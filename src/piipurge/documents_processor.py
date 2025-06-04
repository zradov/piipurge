import os
import io
import re
import fitz
import spacy
import string
import logging
import argparse
import tempfile
import datasets
import unicodedata
from . import consts
from torch import Tensor
from pathlib import Path
from PIL import Image, ImageDraw
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spacy.language import Language
from spacy.pipeline import Sentencizer
from multiprocessing import Pool, Queue
from .utils.drawing import merge_intersecting_rects
from sentence_transformers import SentenceTransformer
from pymupdf import Document, Page, Rect, Pixmap, Point
from .utils.fonts import get_text_length, get_matching_font
from .analyze_images import analyze_print, analyze_handwriting
from .schemas import (
    LinePos, EntitySubstitute, SpanInfo, LineInfo, LineSubstitute,
    SavedDrawingInfo, SavedImageInfo, Paragraph, ImageInfo, RedactInfo, 
    TextRedactInfo, ImageTextInfo
)
from typing import (
    Pattern, Union, Set, List, Tuple, Optional, 
    Dict, Sequence, Generator, Callable
)
from difflib import SequenceMatcher

datasets.disable_progress_bar()
logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)
fitz.TOOLS.set_small_glyph_heights(True)


def find_matching_entities(doc: Doc, regex: Pattern, label: str) -> Doc:
    """
    Finds entities that match the given regex pattern, marks the string
    matches with the Spacy's NER label and assigns named entities to the Doc
    object.

    Args:
        doc: Spacy Doc object
        regex: compiled regex pattern
        label: Spacy's NER label

    Returns: 
        The loaded nlp object.
    """

    matches = regex.finditer(doc.text)
    spans = [doc.char_span(match.start(), match.end(), label=label) for match in matches]
    spans = [s for s in spans if s is not None]
    spans = filter_spans(list(doc.ents) + spans)
    doc.ents = spans
    return doc


@Language.component("ipv4")
def ipv4_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["ipv4"], "IPv4")
    return doc


@Language.component("ipv6")
def ipv6_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["ipv6"], "IPv6")
    return doc


@Language.component("phone")
def phone_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["phone"], "PHONE")
    return doc


@Language.component("email")
def email_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["email"], "EMAIL")
    return doc
    

@Language.component("ssn")
def ssn_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["ssn"], "SSN")
    return doc


@Language.component("medicare")
def medicare_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["medicare"], "MEDICARE")
    return doc

@Language.component("vin")
def vin_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["vin"], "VIN")
    return doc

@Language.component("url")
def url_component(doc: Doc):
    doc = find_matching_entities(doc, consts.PATTERNS["url"], "URL")
    return doc


@Language.component("custom_sentencizer")
def get_sentencizer(doc: Doc):
    sentencizer = Sentencizer(punct_chars=[r"\n"])
    return sentencizer(doc)


def _normalize_utf8_text(text):
    """
    
    """
    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)

    # Replace no-break space with regular space
    text = text.replace('\u00A0', ' ')

    # Translate some special punctuation characters
    translation_table = {
        ord('\u2018'): "'",
        ord('\u2019'): "'",
        ord('\u201C'): '"',
        ord('\u201D'): '"',
        ord('\u2013'): '-',
        ord('\u2014'): '-',
        ord('\u2026'): '...',
        ord('\u00B7'): '-',
    }
    text = text.translate(translation_table)

    # Remove zero-width spaces and formatting chars
    text = re.sub(r'[\u200B\u200E\u200F\uFEFF]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_link(text: str) -> bool:
    """
    Check whether a string refers to an URL or an e-mail address.

    Args:
        text: text to check.

    Returns:
        True if a string represents an URL or e-mail address otherwise False.
    """
    return consts.PATTERNS["url"].match(text) != None or \
        consts.PATTERNS["email"].match(text) != None


def get_spans_similarity(span1: str, span2: str, text_encoder: SentenceTransformer) -> Tensor:
    """
    Measures strings similiarity by computing the cosine similarity.

    Args:
        span1: the first string value.
        span2: the second string value.
        text_encoder: text encoding model.

    Returns:
        a Tensor object containing strings similiarity score.
    """
    span1_embedding = text_encoder.encode(span1, show_progress_bar=False)
    span2_embedding = text_encoder.encode(span2, show_progress_bar=False)
    similarity = text_encoder.similarity(span1_embedding, span2_embedding)

    return similarity


def check_common_acronyms(span_text: str, 
                          common_acronyms: list[str], 
                          text_encoder: SentenceTransformer, 
                          similarity_score_threshold: float=0.95) -> str | None:
    """
    Checks whether the given span text value refers to a common acronym.

    Args:
        span_text: the text the need to be verified against common acronyms.
        common_acronyms: a list of common acronyms strings.
        text_encoder: an text encoder model used for measuring text similarity.
        similarity_score_threshold: threshold used when deciding if two strings are equal or not.


    Returns:
        a string value with matching common acronym if such acronyms is found otherwise None.
    """
    max_similarity = 0
    optim_acronym = None
    for acronym in common_acronyms:
        span_similarity_score = get_spans_similarity(acronym, span_text, text_encoder)
        if (span_similarity_score > max_similarity) and \
                    (span_similarity_score > similarity_score_threshold):
                    optim_acronym = acronym
                    max_similarity = span_similarity_score

    return optim_acronym


def get_most_similar_text(
        texts_to_compare: Optional[List[str]],
        texts: Optional[List[Union[str, Tuple[str, str]]]],
        text_encoder: SentenceTransformer,
        similarity_score_threshold: float = 0.95) -> str | None:
    """
    Finds the most similar text from a list of candidate texts relative to a reference set,
    based on a similarity threshold.

    Args:
        texts_to_compare: list of reference text strings to compare against.
        texts: list of candidate texts to find the most similar match from. 
               Each element can be a string or a tuple of two strings.
        text_encoder: a text encoder model used for computing text embeddings and measuring similarity.
        similarity_score_threshold: minimum similarity score required for a match to be considered.

    Returns:
        A text value (string or tuple) from the `texts` list that is most similar to any of the `texts_to_compare`
        and has a similarity score above the threshold. Returns None if no match meets the threshold.
    """

    most_similar_text = None
    
    if (texts_to_compare is not None) and (texts is not None):
        curr_max_similarity = 0
        for text1 in texts_to_compare:
            for text2 in texts:
                if isinstance(text2, tuple):
                    text_similarity_score = max(
                        get_spans_similarity(text2[0], text1, text_encoder),
                        get_spans_similarity(text2[1], text1, text_encoder))
                else:
                    text_similarity_score = get_spans_similarity(text2, text1, text_encoder)       
                if (text_similarity_score > curr_max_similarity) and \
                    (text_similarity_score > similarity_score_threshold):
                    most_similar_text = text2
                    curr_max_similarity = text_similarity_score

    return most_similar_text


def get_ent_replacement(ent: Span,  
                        suffix_type: str | int, 
                        entities_count: int=0) -> str:
    """
    For the given Spacy named entity type returns its unique identifier in the document.
    
    Args:
        ent: the target Span object
        suffix_type: a suffix value assigned to the span replacement string, 
                     it can be integer or character and it represents an index 
                     value of the entity object in the document.
        entities_count: current count of entities with type equal to the the ent.label_ type,
                        it is used to determine the next index value.

    Returns:
        a string representing the entity replacement string.             

    Raises:
        Exception: if the suffix_type is of unsupported type.
    """
    if suffix_type is str:
        suffix = int(round(entities_count / len(string.ascii_uppercase))) * "A" + \
            string.ascii_uppercase[entities_count % len(string.ascii_uppercase)]
        return f"\"{consts.ENTITY_DESC[ent.label_][0]} {suffix}\""
    elif suffix_type is int:
        return f"\"{consts.ENTITY_DESC[ent.label_][0]} {entities_count+1}\""
    
    raise Exception(f"Unsupported suffix type '{suffix_type}")
             

def get_closest_ent_name(texts: List[str],
    ent_cat_subs: Set[str],
    relations: List[Tuple[str, str]],
    text_encoder: SentenceTransformer) -> Optional[str]:
    """
    Finds the closest entity name from a set of candidate entities based on text similarity.
    The function first checks relations for matching entities, then falls back to similarity search.

    Args:
        texts: List of input text strings to compare against.
        ent_cat_subs: Set or list of candidate entity names to match.
        relations: List of relation tuples (e.g., [(ent1, ent2), ...]) to check for entity matches.
        text_encoder: Text encoding model (e.g., SentenceTransformer) for similarity computation.

    Returns:
        The closest matching entity name (str) if found, otherwise None.
    """
    relation = get_most_similar_text(texts, relations, text_encoder)
    closest_ent_name = None

    if relation:
        closest_ent_name = relation[0] if relation[0] in ent_cat_subs \
            else relation[1] if relation[1] in ent_cat_subs \
            else None
        if not closest_ent_name:
            closest_ent_name = get_most_similar_text(texts, ent_cat_subs, text_encoder)

    return closest_ent_name


def _update_subs(subs: Dict[str, Dict[str, EntitySubstitute]], 
                ents: List[Span], 
                relations: List[Tuple[str, str]], 
                common_acronyms: List[str], 
                text_encoder: SentenceTransformer, 
                paragraph: Paragraph) -> None:
    """
    Updates the dictionary containing substitute strings for recognized 
    named entities in the document.

    Args:
        subs: dictionary with substitute strings for recognized named entities in the document.
              For details info about the data structure check the _process_org_entity or the 
              _process_standard_entity function.
        ents: recognized named entities in the document.
        relations: a list, where each element represents a combination of an acronym and its longer form.
        common_acronyms: a list of strings representing acronyms.
        text_encoder: the model for the text encoding.
        paragraph_index: an index of the paragraph to which the named entity object belongs.

    Returns:
        None
    """
    for ent in ents:
        if not _should_process_entity(ent, common_acronyms, text_encoder):
            continue

        ent_subs = subs.get(ent.label_, {})
        suffix_type = consts.ENTITY_DESC[ent.label_][1]
            
        if ent.label_ == "ORG":
            _process_org_entity(ent, relations, ent_subs, text_encoder, suffix_type, paragraph)
        else:
            _process_standard_entity(ent, ent_subs, suffix_type, paragraph)
            
        subs[ent.label_] = ent_subs


def _should_process_entity(ent: Span, 
                           common_acronyms: List[str], 
                           text_encoder: SentenceTransformer):
    """
    Determines whether the text representing a named entity should be further processed.

    Args:
        ent: named entity object
        common_acronyms: a list of strings representing acronyms
        text_encoder: the model for the text encoding
        
    Returns:
        True if the entity type is in the supported entity types dictionary otherwise False
    """
    if ent.label_ not in consts.ENTITY_DESC:
        return False
    if ent.label_ == "ORG":
        acronym = check_common_acronyms(ent.text, common_acronyms, text_encoder)
        if acronym:
            return False
        
    return True


def _process_org_entity(ent: Span, 
                        relations: List[Tuple[str, str]], 
                        ent_subs: dict[str, List[EntitySubstitute]],
                        text_encoder: SentenceTransformer,
                        suffix_type: str,
                        paragraph: Paragraph) -> None:
    """
    Determines replacement string for the ORG named entity.

    Args:
        ent: named entity object
        relations: a list, where each element represents a combination of an acronym and its longer form.
        ent_subs: a dictionary of all string values, related to the recognized named entities, in a document 
                  and their replacement strings where the tuple value is in the following format:
                  (REPLACEMENT_STRING, STARTING_CHAR_IN_DOCUMENT, ENDING_CHAR_IN_DOCUMENT, 
                   STARTING_CHAR_IN_SENTENCE, ENDING_CHAR_IN_SENTENCE, PARAGRAPH_INDEX)
        text_encoder: the model for the text encoding
        suffix_type: a suffix value assigned to the span replacement string, 
                     it can be integer or character and it represents an index 
                     value of the entity object in the document.
        paragraph_index: an index of the paragraph to which the named entity object belongs.

    Returns:
        None
    """
    texts_to_compare = [ent.text]
    relation = get_most_similar_text([ent.text], relations, text_encoder)
    if relation:
        texts_to_compare.extend([i for i in relation])
    temp_relations = (relations if relations else []) + list(ent_subs.keys())
    closest_key_name = get_closest_ent_name(texts_to_compare, ent_subs, temp_relations, text_encoder)
    
    if closest_key_name:
        ent_replacements = ent_subs[closest_key_name]
        _ = [ent_replacements.append(s) 
             for s in _create_entity_substitute(ent, ent_subs[closest_key_name][0][0], paragraph)]
        return

    ent_subs[ent.text] = list(_create_entity_substitute(
        ent, get_ent_replacement(ent, suffix_type, len(ent_subs)), paragraph))


def _find_best_matching_string(source_text: str, 
                              target_text: str, 
                              start_char: int,
                              end_char: int,
                              best_match_ratio: float,
                              iterate_fn: Callable[[int, int], Tuple[int, int]]) -> Tuple[float, int, int]:
    sm = SequenceMatcher(None, source_text, target_text[start_char:end_char])
    old_start_char, old_end_char = start_char, end_char
    while True:
        if sm.ratio() < best_match_ratio:
            break
        best_match_ratio = sm.ratio()
        old_start_char, old_end_char = start_char, end_char
        start_char, end_char = iterate_fn(start_char, end_char)
        sm.set_seq2(target_text[start_char:end_char])

    return best_match_ratio, old_start_char, old_end_char
        

def _find_entity_boundaries(ent: Span, paragraph: Paragraph) -> Tuple[int, int, float]:
    text_index = paragraph["text"].find(ent.text)
    if text_index != -1:
        return text_index, text_index + len(ent.text), 1.0
    char_start_index = _normalize_utf8_text(paragraph["text"]).index(ent.text)
    start_char = _normalize_utf8_text(paragraph["text"]).index(ent.text)
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        ent.text, paragraph["text"], char_start_index, 
        char_start_index + len(ent.text), 0.0, lambda s, e: (s + 1, e + 1),
    )
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        paragraph["text"][start_char:end_char], paragraph["text"], start_char, 
        start_char + len(ent.text), best_match_ratio, lambda s, e: (s, e + 1),
    )
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        paragraph["text"][start_char:end_char], paragraph["text"], start_char, 
        end_char, best_match_ratio, lambda s, e: (s - 1, e),
    )
    logging.info(f"(_find_entity_boundaries): best_match_ratio: {best_match_ratio}")

    return start_char, end_char, best_match_ratio


def _process_standard_entity(ent: Span, 
                             ent_subs: dict[str, List[EntitySubstitute]],
                             suffix_type: str,
                             paragraph: Paragraph) -> None:
    """
    Determines replacement string for the named entities other than the ORG entity.

    Args:
        ent: named entity object
        ent_subs: a dictionary of all named entities in a document and their replacement strings
                  where the tuple value is in the following format:
                  (REPLACEMENT_STRING, STARTING_CHAR_IN_DOCUMENT, ENDING_CHAR_IN_DOCUMENT, 
                   STARTING_CHAR_IN_SENTENCE, ENDING_CHAR_IN_SENTENCE, PARAGRAPH_INDEX)
        suffix_type: a suffix value assigned to the span replacement string, 
                     it can be integer or character and it represents an index 
                     value of the entity object in the document.
        paragraph_index: an index of the paragraph to which the named entity object belongs.

    Returns:
        None
    """

    if ent.text in ent_subs:
        replacements = ent_subs[ent.text]
        # Pick the first replacement item for the entity name and select its replacement string
        _ = [replacements.append(s) 
             for s in _create_entity_substitute(ent, ent_subs[ent.text][0][0], paragraph)]
    else:
        ent_subs[ent.text] = list(_create_entity_substitute(
            ent, 
            get_ent_replacement(ent, suffix_type, len(ent_subs)), 
            paragraph))


def _is_entity_inside_line(ent_start_char: int, ent_end_char) -> bool:
    def fn(line):
        return (ent_start_char >= line.line_start_char and ent_start_char <= line.line_start_char) or \
            (ent_end_char >= line.line_start_char and ent_end_char <= line.line_end_char)

    return fn


def _get_paragraph_line_indices(ent: Span, paragraph: Paragraph, min_match_ratio=0.95) -> List[int]:
    ent_start_char, ent_end_char, best_match_ratio = _find_entity_boundaries(ent, paragraph)
    if best_match_ratio < min_match_ratio:
        raise f"Text {ent.text} not found in the paragraph {paragraph['text']}"
    lines_indices = list(filter(_is_entity_inside_line(ent_start_char, ent_end_char), paragraph["lines"]))

    return lines_indices


def _create_entity_substitute(ent: Span, new_val: str, paragraph: Paragraph, min_match_ratio=0.95) \
    -> Generator[EntitySubstitute, None, None]:
    ent_start_char, ent_end_char, best_match_ratio = _find_entity_boundaries(ent, paragraph)
    if best_match_ratio < min_match_ratio:
        raise f"Text {ent.text} not found in the paragraph {paragraph['text']}"
    lines = list(filter(_is_entity_inside_line(ent_start_char, ent_end_char), paragraph["lines"]))
    
    return (
        EntitySubstitute(
            old_val=paragraph["text"][ent_start_char:ent_end_char],
            new_val=new_val,
            start_char=ent_start_char,
            end_char=ent_start_char + len(ent.text),
            sent_start_char=max(ent_start_char-line.line_start_char, line.line_start_char),
            sent_end_char=min(ent_end_char-line.line_start_char, line.line_end_char),
            line_index=line.line_index,
            paragraph_index=paragraph["index"],
            ent_type=ent.label_
    ) for line in lines)


def is_acronym(short_form: str, long_form_entities: List[str]) -> bool:
    """
    Performs a simple check to see if the first characters of each named entity object 
    combined, in the list of named entity objects, and the given acronym are related.

    Args:
        short_form: a string representing a possible acronym.
        long_form_entities: a list of Span objects representing named entities

    Returns:
        True if the short string form and entities list relate to the same acronym
        otherwise False.
    """
    long_acronym = ''.join(ent.text[0].upper() for ent in long_form_entities)
    long_acronym = re.sub(r"[^\w]", "", long_acronym)
    return short_form.upper() == long_acronym


def _find_short_long_relations(acronyms_ents: Sequence[Span]) -> List[Tuple[str, str]]:
    """
    Finds acronyms and their expanded form.

    Args:
        acronyms_ents: a sequence of named entities recognized by the acronym identification model.
    
    Returns:
        a list where each element contains the expanded acronym form and the acronym. 
    """
    if len(acronyms_ents) == 0:
        return
    long_form = []
    long_form_range = (0, 0)
    relations = []
    
    for ent in acronyms_ents:
        tag, _, start, end = ent.label_, ent.text, ent.start, ent.end
        if _is_new_long_form(tag, start, long_form_range):
            long_form = [(ent)]
            long_form_range = (start, end)
        elif _is_continuation_of_long_form(tag, start, long_form_range):
            long_form.append(ent)
            long_form_range = (long_form_range[0], end)
        elif _is_short_form(tag, start, long_form, long_form_range, ent):
            relations.append((" ".join([e.text for e in long_form]), ent.text))
            long_form_range = (0, 0)

    return relations


def _is_new_long_form(tag: str, start: int, long_form_range: Tuple[int, int]) -> bool:
    """
    Checks if the given IOB format tag refers to a beginning of an expanded acronym.

    Args:
        tag: IOB format tag e.g. "B-long", "B-short", "I-long", "I-short", "O",
             for details check https://huggingface.co/datasets/amirveyseh/acronym_identification
        start: starting index of the token in the text.
        long_form_range: starting and ending index of tokens forming a possible exanded acronym.

    Returns:
        True if the tag marks the beginning of a chunk of text, is inside a chunk of text or if 
        the token's distance between the last recognized IOB token and the current token index is 
        greater than 2, otherwise False.
    """
    return (tag.startswith("B-long") or tag.startswith("I-long")) and \
            start - long_form_range[1] > 2


def _is_continuation_of_long_form(tag: str, start: int, long_form_range: Tuple[int, int]) -> bool:
    """
    Checks if the given IOB format tag is inside an expanded acronym.

    Args:
        tag: IOB format tag e.g. "B-long", "B-short", "I-long", "I-short", "O",
             for details check https://huggingface.co/datasets/amirveyseh/acronym_identification
        start: starting index of the token in the text.
        long_form_range: starting and ending index of tokens forming a possible exanded acronym.

    Returns:
        True if a token with the given tag is inside a set of tokens representing an expanded 
        acronym otherwise False.
    """
    return (tag.startswith("I-long")) or \
           (tag.startswith("B-long") and \
           (start - long_form_range[1] <= 2))


def _is_short_form(tag: str, 
                   start: int, 
                   long_form: List[Span], 
                   long_form_range: Tuple[int, int], 
                   ent: Span) -> bool:
    """
    Checks if the named entity with the given IOB format tag is actually an acronym.

    Args:
        tag: IOB format tag e.g. "B-long", "B-short", "I-long", "I-short", "O",
             for details check https://huggingface.co/datasets/amirveyseh/acronym_identification
        start: starting index of the token in the text.
        long_form: a list of tokens forming an expanded acronym.
        long_form_range: starting and ending index of tokens forming a possible exanded acronym.
        ent: a named entity object representing a possible acronym.

    Returns:
        True if the named entity object represents an acronym otherwise False.
    """
    if (not tag.startswith("B-short") or
        not long_form or 
        (start - long_form_range[1]) > 2):
        return False
    
    return (is_acronym(ent.text, long_form) or 
            (start - long_form_range[1] <= 2 and 
             long_form[-1].sent == ent.sent))


def get_redacted_text(line_subs: List[LineSubstitute], 
                      redact_char="x") -> str:
    """
    It replaces the redacted text with the specified replacement character and returns
    the redacted text.

    Args:
        line_subs: a list of ListSubstitute objects.
        redact_rect: a redacted text replacement character.

    Returns:
        text with redacted text being replaced with the replacement character.
    """
    redacted_text = line_subs[0].line_info.text
    sorted_line_subs = list(sorted(line_subs, key=lambda i: i.ent_start_char, reverse=True))

    for sub in sorted_line_subs:
        redacted_text = redact_text(sub, redacted_text, redact_char)

    return redacted_text


def redact_text(sub: LineSubstitute, 
                redacted_text: str, 
                redact_char: str) -> str:
    """
        
    """
    # It's assumed that all spans in a line have the same font name and the font size.
    # For more robust solution each span's font name and font size should be inspected
    # separately. 
    font = sub.line_info.spans[0]["font"]
    size = sub.line_info.spans[0]["size"]
    
    new_val = len(sub.old_val) * redact_char
    while get_text_length(new_val, font_name=font, font_size=size) >= \
        get_text_length(sub.old_val, font_name=font, font_size=size):
        new_val = new_val[:-1]
    text = redacted_text[:sub.ent_start_char] + \
        new_val + redacted_text[sub.ent_end_char:]
    
    return text


def get_redact_annots(page: Page, 
                      line_subs: LineSubstitute, 
                      use_span_bg: bool=False, 
                      fill_color: Tuple[float, float, float]=(0, 0, 0)) \
                      -> List[TextRedactInfo]:
    """
    Returns a list of redaction information.

    Args:
        page: PyMuPDF's page object.
        line_subs: a list of all text substitutes for a specific line.
        use_span_bg: whether or not to use span background as the redaction rectangle fill color.
        fill_color: default fill color for the redaction rectangle.
    """
    redactions = []

    if line_subs:
        for line_sub in line_subs:
            rects = page.search_for(line_sub.old_val, clip=line_sub.line_info.spans_bbox, quads=False, flags=0)
            for r in rects:
                redact_info = get_text_redact_info(page.number, r, line_sub, use_span_bg, fill_color)
                redactions.append(redact_info)
    
    return redactions


def _get_redact_rect(rect: Rect, offset_y: float=0.01) -> Tuple[float, float, float, float]:
    center_y = rect[1] + (rect[3] - rect[1]) / 2
    redact_rect = (rect[0], center_y - offset_y, rect[2], center_y + offset_y)
    
    return redact_rect


def get_text_redact_info(
        page_number: int,
        rect: Rect,
        line_sub: LineSubstitute,
        use_span_bg: bool,
        default_redact_color: Tuple[float, float, float]) -> TextRedactInfo:
    """
    Return information about the text redaction area.

    Args:
        page_number: the page's index in the PDF document.
        rect: the redaction area.
        line_sub: information about all text replacements in a specific line.
        use_span_bg: whether or not to use span background as the redaction rectangle fill color.
        default_redacted_color: default fill color for the redaction rectangle.

    Returns:
        a text redaction info.
    """
    draw_redact_rect = [rect[0], line_sub.line_info.spans_bbox[1], rect[2], line_sub.line_info.spans_bbox[3]]
    # The height of the redaction rectangle will be reduced as much as possible, in order to 
    # avoid the issue with redactions of text in the neighboring lines, as describe in the issue
    # https://github.com/pymupdf/PyMuPDF/discussions/1810
    redact_rect = _get_redact_rect(rect)
    # If span background should be used as the fill color, for the redact annotation rectangle,
    # choose the fill color of the first span containing text that need to be redacted.
    redact_color = line_sub.sub_spans[0]["fill_color"] if use_span_bg else default_redact_color
    
    return TextRedactInfo(
        page_number=page_number,
        rect=redact_rect,
        draw_rect=draw_redact_rect,
        line_sub=line_sub, 
        fill_color=redact_color
    )


def fill_bg(page: Page, 
            line: LineInfo, 
            margin: int=0) -> None:
    """
    Fills in the background of all spans containing the text that's being redacted.

    Args:
        page: PyMuPDF's page object.
        line: a LineInfo object.
        margin: margin of the redaction bounding box.

    Returns:
        None
    """

    for span in line.spans:
        bbox = span["bbox"]
        redact_bbox = [bbox[0]-margin, bbox[1]-margin, bbox[2]+margin, bbox[3]+margin]
        page.draw_rect(redact_bbox, color=span["fill_color"], fill=span["fill_color"])


def replace_text(page: Page, 
                 line: LineInfo, 
                 text: str) -> None:
    """
    Replaces the text in spans objects containing the text that's being redacted.

    Args:
        page: a PyMuPDF's Page object.
        line: a metadata object about the line containing text that's being redacted.
        text: replacement text

    Returns:
        None
    """
    
    offset = 0
    for span in line.spans:
        text_color = span["color"] if isinstance(span["color"], tuple) \
            else [round(c/255.0, 2) for c in fitz.sRGB_to_rgb(span["color"])]
        kwargs = dict(fontsize=span["size"], 
                      color=text_color, 
                      fill_opacity=span["alpha"])
        
        font_metadata = get_matching_font(span["font"])
        kwargs["fontfile"] = font_metadata["font_path"]
        span_text = text[span["text_start"]:span["text_end"]]
        _ = page.insert_text((span["origin"][0] + offset, span["origin"][1]), span_text, **kwargs)
        new_text_length_pts = get_text_length(span_text, font_name=font_metadata["full_name"], font_size=span["size"])
        offset += new_text_length_pts - (span["span_bbox"][2] - span["span_bbox"][0])


def is_neighboring_line(line: LineInfo, 
                         lines: List[int]) -> bool:
    """
    Checks if the specified line is a neighboring line for any of the lines in the list.

    Args:
        line: a LineInfo object representing a paragraph line for which we need to check if it
              is a neighboring line.

    Returns:
        True if the specified line is a neighboring line to one of the lines in the list, otherwise False.
    """

    for line_idx in lines:
        if abs(line.paragraph_line_index - line_idx) == 1:
            return True
        
    return False


def get_redacted_paragraphs(annotations: List[TextRedactInfo]) \
    -> Dict[Tuple[int, int], List[Tuple[Rect, LineSubstitute]]]:
    """
    Groups redaction rectangles and information about the redacted text, in a line, by the page number and the paragraph index.

    Args:
        annotations: a list of text redaction annotations.

    Returns:
        a dictionary object where keys are tuples of (page_number, paragraph_index) and values lists of tuples 
        (redact_rect, line_substitute)
    """

    redacted_paragraphs = {}

    for annot in annotations:
        redact_rect = annot["rect"]
        line_sub = annot["line_sub"]
        key = (line_sub.line_info.page_number, line_sub.entity_substitute.paragraph_index)
        parag_replacements = redacted_paragraphs.get(key, [])
        parag_replacements.append((redact_rect, line_sub))
        redacted_paragraphs[key] = parag_replacements

    return redacted_paragraphs


def draw_paragraph_annots(page: Page, 
                          line: LineInfo,
                          redacted_lines: List[int], 
                          parag_replacements: List[Tuple[Tuple[float, float, float, float], LineSubstitute]], 
                          redact_rect_color: Tuple[float, float, float]=(0, 0, 0)) -> None:
    """
    Draws a separate redaction annotation rectangle over each redacted text area.

    Args:
        page: PyMuPDF's page object.
        line: a LineInfo object.
        redacted_lines: a list of integers representing indices of lines in a paragraph.
        parag_replacements: a list of tuples consisting of rectangle of redacted area and the LineSubstitute object containing metadata
                            about the line and the text being redacted. 
        redact_rect_color: a fill in color of the redaction rectangle.
                            
    Returns:
        None
    """

    if line["paragraph_line_index"] in redacted_lines:
        line_subs = [r for r in parag_replacements 
                        if r[1]["paragraph_line_index"] == line["paragraph_line_index"]]
        for redact_rect, _ in line_subs:
            page.draw_rect(redact_rect,
                           color=redact_rect_color,
                           fill=redact_rect_color)
            

def _reconstruct_deleted_text(pdf: Document, 
                             redactions: List[TextRedactInfo], 
                             paragraphs: List[Paragraph], 
                             draw_redact_annot: bool=False) -> None:
    """
    Recontructs deleted text caused by the redaction of text in the overlapping lines and, if enabled, 
    draws redaction annotation rectangle.
    More about this issue could be found at link: https://github.com/pymupdf/PyMuPDF/discussions/1810

    Args:
        pdf: a PyMuPDF document object.
        redactions: a list of text redaction metadata.
        paragraphs: a list of paragraphs.
        draw_redact_annot: whether or not to draw a redaction rectangle.

    Returns:
        None
    """

    redacted_paragraphs = get_redacted_paragraphs(redactions)

    for (page_num, paragraph_index), parag_replacements in redacted_paragraphs.items():
        redacted_lines = [line_substitute.line_info.paragraph_line_index 
                          for (_, line_substitute) in parag_replacements]

        for line_info in paragraphs[paragraph_index]["lines"]:
            if line_info.paragraph_line_index in redacted_lines or \
                is_neighboring_line(line_info, redacted_lines):
                if line_info.paragraph_line_index in redacted_lines:
                    text = get_redacted_text([line_substitute
                                              for (_, line_substitute) in parag_replacements 
                                              if line_substitute.line_info.paragraph_line_index == line_info.paragraph_line_index])
                else:
                    text = line_info.text

                fill_bg(pdf[page_num], line_info, margin=0.2)                
                replace_text(pdf[page_num], line_info, text)
                if draw_redact_annot:
                    draw_paragraph_annots(pdf[page_num], line_info, redacted_lines, parag_replacements)
                

def substitute_page_entities(pdf: Document, 
                             lines_subs: Dict[LinePos, List[Dict]], 
                             min_distance=3) -> List[TextRedactInfo]:
    """
    Returns a list of TextRedactInfo objects related to the redaction of textual information.

    Args:
        pdf: a PDF document object.
        lines_subs: a dictionary containing pairs of line positions and all text replacements for each line.
                    For details about the data structure see the documentation for the _map_subs_to_lines function,
                    specifically description for the return value.
        min_distance: minimum distance between the location of the ending character, of the previous entity instance,
                      and the starting character location for the new entity instance. If the value is greater than
                      the specified the new entity instance will be considered as a separate instance of the particular
                      entity type, otherwise the location of the ending character, of the previous entity instance, will
                      be expand to match the ending character of the new instance.
                    
    Returns:
        a list of redaction annotations for each recognized named entity in the document.
    """

    redactions = []

    for (page_num, _), occurences in lines_subs.items():
        replacements = []
        orgs_to_replace = []
    
        for line_subs in occurences:
            if line_subs.entity_substitute.ent_type == "ORG":
                if orgs_to_replace:
                    last_org = orgs_to_replace[-1]
                    if line_subs.old_val == last_org.old_val:
                        if (line_subs.ent_start_char - last_org.ent_end_char <= min_distance) and \
                            (last_org.entity_substitute.new_val == line_subs.entity_substitute.new_val):
                            # The new ORG will not be added, instead the previous ORG's ending character index 
                            # will be expanded so that it includes the new ORG. 
                            new_org_end_char = line_subs.ent_end_char
                            if (len(last_org.line_info.text) - 1 < new_org_end_char) and \
                               (last_org.line_info.text[new_org_end_char + 1] == ")"):
                                new_org_end_char += 1
                            last_org = last_org._replace(ent_end_char=new_org_end_char)
                            orgs_to_replace[-1] = last_org
                            continue
                orgs_to_replace.append(line_subs)
            else:
                replacements.append(line_subs)

        for org_line_sub in orgs_to_replace:
            replacements.append(org_line_sub)

        page = pdf[page_num]

        if replacements:
            annots = get_redact_annots(page, replacements)
            if annots:
                redactions.extend(annots)

    return redactions


def add_utf8_normalization(nlp: Language):
    original_make_doc = nlp.make_doc
    
    def make_doc_with_utf8_normalization(text):
        normalized_text = _normalize_utf8_text(text)
        return original_make_doc(normalized_text)
    
    nlp.make_doc = make_doc_with_utf8_normalization


def load_nlp_model(model_name: str="en_core_web_trf") -> Language:
    """
    Loads and returns a NLP model used for PDF text processing.

    Args:
        model_name: a name of the model

    Returns:
        a Spacy's text-processing pipeline object.
    """

    nlp = spacy.load(model_name)
    add_utf8_normalization(nlp)
    components = ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe('sentencizer')
    for c in components:
        nlp.add_pipe(c, last=True)

    return nlp

 
def load_nlp_acronyms_model() -> Language:
    """
    Loads and returns a NLP model used specifically for acronym identification.

    Args:
        model_name: a name of the model

    Returns:
        a Spacy's text-processing pipeline object.
    """

    nlp_acronyms = spacy.load(consts.ACRONYMS_MODEL_DIR)
    nlp_acronyms.add_pipe('sentencizer')

    return nlp_acronyms


def run_nlp(paragraphs: List[Paragraph]) -> Dict[LinePos, List[Dict]]:
    """
    Processes text paragraphs and returns a dictionary object containing information 
    about lines containing text substitutes and their replacement values.
    
    Args:
        paragraphs: a paragraphs list

    Returns:
        a dictionary containing pairs of line positions and all text replacements for each line.
        For details about the data structure see the documentation for the _map_subs_to_lines function,
        specifically description for the return value.
    """

    nlp = load_nlp_model()
    nlp_acronyms = load_nlp_acronyms_model()
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  

    subs = {} 
    for paragraph in paragraphs:
        # Spacy 
        doc = nlp(paragraph["text"])
        acronyms_doc = nlp_acronyms(paragraph["text"])

        relations = _find_short_long_relations(acronyms_doc.ents)

        for ent in doc.ents:
            logging.debug(f"ent.text: {ent.text}, ent.label_: {ent.label_}")

        _update_subs(subs, doc.ents, relations, consts.COMMON_ACRONYMS, encoder_model, paragraph)

    lines_subs = _map_subs_to_lines(subs, paragraphs)

    return lines_subs


def _get_pdf_lines(pdf: Document) -> Dict[str, List[LineInfo]]:
    """
    Finds and returns all spans in the PDF document.

    Args:
        pdf: a PyMuPDF document object.

    Returns:
        a dictionary of all spans found in the page where key refers to a specific line 
        in the text and values refer to a list of spans found in the specific line.
    """

    pdf_lines = {}
    
    for page in pdf:    
        lines = get_pdf_page_lines(page)
        for line, line_infos in lines.items():
            if line in pdf_lines:
                pdf_lines[line].extend(line_infos)
            else:
                pdf_lines[line] = line_infos

    return pdf_lines


def get_pdf_span_boundaries(span: Dict) -> Rect:
    """
    Calculates the exact bounding box boundaries of a span so that the original span 
    bounding box will shrink to the height and width of the text content.

    Args:
        span: a span object

    Returns:
        a bounding box boundaries matching exactly the content of the span
    """

    a = span["ascender"]
    d = span["descender"]
    r = fitz.Rect(span["bbox"])
    o = fitz.Point(span["origin"])
    r.y1 = o.y - span["size"] * d / (a - d)
    r.y0 = r.y1 - span["size"]

    return r


def get_pdf_page_lines(page: Page) -> Dict[str, List[LineInfo]]:
    """
    Finds and returns all lines in the page.

    Args:
        page: a PyMuPDF page object.

    Returns:
         a dictionary of all lines found in the page where key refers to a specific line 
         in the text and values refer to a list of spans found in the specific line.
    """
    
    pdf_spans = {}
    blocks = page.get_text("dict")["blocks"]
    zoom_level = 3
    zoom_matrix = fitz.Matrix(zoom_level, zoom_level)
    page_pixmap = page.get_pixmap(matrix=zoom_matrix)
    
    for block_idx, block in enumerate(blocks):
        _process_text_block(pdf_spans, page.number, block, block_idx, page_pixmap, zoom_level)

    return pdf_spans


def _process_text_block(pdf_spans: Dict[str, List[SpanInfo]],
                        page_number: int,
                        block: Dict, 
                        block_idx: int,
                        page_pixmap: Pixmap,
                        zoom_level: int) -> None:
    """
    Processes a page’s text block and updates the dict object containing all 
    PDF spans found in the page.

    Args:
        pdf_spans: a dictionary of all spans found in the page where key refers
                   to a specific line in the text and values refer to a list of 
                   spans found in the specific line.
        page_number: the page in the PDF document.
        block: a page’s text block.
        block_idx: an index of the text block relative to all text blocks in the page.
        page_pixmap: a rendered representation of the PDF document's page, used to 
                     determine the exact color of the spans' background.
        zoom_level: the zoom level used when creating the Pixmap object. It's used to 
                    scale line's bound box PDF coordinates to match the higher resolution 
                    space of the image representing the rendered PDF Page.

    Returns:
        None
    """

    is_text_block = block["type"] == 0

    if is_text_block:
        for line_idx, line in enumerate(block["lines"]):
            line_text = ""
            spans = []
            if len(line["spans"]):
                # Scale the bounding box coordinates from PDF space to the 
                # image space by multiplying coordinates by the zoom factor. 
                fill_color = tuple(c/255.0 for c in page_pixmap.pixel(
                    int(line["bbox"][0] * zoom_level), 
                    int(line["bbox"][1] * zoom_level)))

            for span in line["spans"]:
                span_info = _get_span_info(span, fill_color, text_start=len(line_text))
                line_text += span_info["text"]
                # For spans containing only whitespace characters the 'size' key won't be set.
                #if "size" in span_info:
                spans.append(span_info)
            if line_text != "":
                line_info = _get_line_info(page_number, block_idx, line_idx, line_text, line["bbox"], spans)
                occurences = pdf_spans.get(line_text, [])
                occurences.extend([line_info])
                pdf_spans[line_text] = occurences


def _get_span_info(span: dict, 
                  fill_color: Tuple[float, float, float], 
                  text_start: int) -> SpanInfo:
    """
    Creates and returns a SpanInfo wrapper for the span dictionary.

    Args:
        span: a span dictionary.
        fill_color: span background color.
        text_start: position of the first character in the span relative to the 
                    span's position in the line that contains the span measured 
                    in number of characters.
        
    Returns:
        a SpanInfo object.
    """

    span_text = span["text"]

    #if span_text.strip() == "":
    #    return SpanInfo(text=span_text)

    span_bbox = get_pdf_span_boundaries(span)            
    span_info = SpanInfo(
        size=span["size"],
        font=span["font"],
        color=span["color"],
        fill_color=fill_color,
        alpha=span["alpha"],
        origin=span["origin"],
        bbox=span["bbox"],
        span_bbox=span_bbox,
        # Use original text instead of the whitespaces the stripped text 
        # will be used to match text of sentences returned by the spaCy model.
        text=span_text,
        text_start=text_start,
        text_end=text_start + len(span_text)
    )

    return span_info


def _get_line_info(page_number: int, 
                  block_idx: int, 
                  line_idx: int,
                  line_text: str,
                  line_bbox: Tuple[float, float, float],
                  spans: List[SpanInfo]) -> LineInfo:
    """
    Creates and returns a LineInfo object.

    Args:
        page_number: index of the page containing the line.
        block_idx: a PyMuPDF's Page text block index.
        line_idx: an index of the line within the text block.
        line_text: text of the line.
        line_bbox: line's bounding box boundaries.
        spans: a list of all spans found in the line.

    Returns:
        a LineInfo object
    """
    
    spans_max_y = max([s["span_bbox"][-1] for s in spans])
    line_info = LineInfo(
        page_number=page_number,
        block_index=block_idx,
        line_index=line_idx,
        line_bbox=line_bbox,
        spans_bbox=((spans[0]["span_bbox"][0], spans[0]["span_bbox"][1], spans[-1]["span_bbox"][-2], spans_max_y)),
        text=line_text,
        spans=spans
    )

    return line_info


def get_paragraphs(pdf: Document, line_gap_threshold: int=5) -> List[Paragraph]:
    """
    Returns a list of text paragraphs.

    Args:
        pdf: a PyMuPDF Document object.
        line_gap_threshold: a minimum gap required, between neighboring lines, in 
                            order to consider the lines as part of the same paragraph.

    Returns:
        a list of Paragraph objects.
    """

    sorted_line_infos = _get_sorted_pdf_lines(pdf)
    paragraphs = _group_lines_into_paragraphs(sorted_line_infos, line_gap_threshold)  

    return paragraphs  


def _get_sorted_pdf_lines(pdf: Document) -> List[LineInfo]:
    """
    Extracts text lines from the PDF documents and returns them 
    sorted by the page number and the spans bounding box in ascending
    order.

    Args:
        pdf: a PyMuPDF Document object.

    Returns:
        a list of sorted PDF document text lines. 
    """

    pdf_lines = _get_pdf_lines(pdf)
    all_lines = []

    for line_infos in pdf_lines.values():
        all_lines.extend(line_infos)

    return sorted(all_lines, key=lambda l: (l.page_number, l.spans_bbox[-1]))


def _group_lines_into_paragraphs(
        line_infos: List[LineInfo],
        line_gap_threshold: int) -> List[Paragraph]:
    """
    Groups lines into paragraphs.

    Args:
        line_infos: a list of LineInfo metadata objects.
        line_gap_threshold: a minimum gap required, between neighboring lines, in 
                            order to consider the lines as part of the same paragraph.

    Returns:
        a list of paragraphs.
    """

    paragraphs_groups = []
    current_paragraph = []
    last_y = None
    char_offset = 0
    paragraph_line_index = 0
    current_page_num = 0

    for line_info in line_infos:
        _, y0, _, y1 = line_info.spans_bbox

        if (last_y is not None and (y0 - last_y) > line_gap_threshold) or \
            line_info.page_number != current_page_num:
                if current_paragraph:
                    paragraphs_groups.append(current_paragraph)
                current_paragraph = []
                char_offset = 0
                paragraph_line_index = 0
                current_page_num = line_info.page_number

        temp_line_info = line_info._replace(line_start_char=char_offset, 
                                            line_end_char=char_offset + len(line_info.text),
                                            paragraph_line_index = paragraph_line_index)
        current_paragraph.append(temp_line_info)
        # Each line will be separated by a whitespace character.
        char_offset += len(temp_line_info.text) + 1
        last_y = y1
        paragraph_line_index += 1

    if current_paragraph:
        paragraphs_groups.append(current_paragraph)

    return [_create_paragraph(idx, group) for idx, group in enumerate(paragraphs_groups)]


def _create_paragraph(index: int, lines: List[LineInfo]) -> Paragraph:
    text = " ".join([line_info.text for line_info in lines])
    
    return Paragraph(
        index=index,
        text=text,
        lines=lines,
        page_number=lines[0].page_number
    )


def _map_subs_to_lines(subs: Dict[str, Dict[str, List[EntitySubstitute]]], 
                      paragraphs: List[Paragraph]) -> Dict[LinePos, List[Dict]]:
    """
    Maps text substitutes to specific lines in paragraphs.

    Args:
        subs: a dictionary of all recognized named entities types in the document along with their 
              occurences in the document. For details about the data structure of values in the list
              of named entities occurences see either the _process_org_entity or the _process_standard_entity
              function.
        paragraphs: a list of all paragraphs in the document.
              
    Returns:
        a dictionary where key is a tuple of the page number and the line's bound box boundaries while the value
        is a list of dictionaries containing information about the text substitutes for that particular line. 
    """

    lines_subs = {}

    for label_type in subs:
        for old_val, entity_substitutes in subs[label_type].items():
            for entity_substitute in entity_substitutes:
                for line_info in paragraphs[entity_substitute.paragraph_index]["lines"]:
                    if line_info.line_start_char > entity_substitute.end_char - 1:
                        break
                    if (entity_substitute.start_char >= line_info.line_start_char) and \
                        (entity_substitute.end_char <= line_info.line_end_char):
                        line_subs = lines_subs.get((line_info.page_number, line_info.line_bbox), [])
                        line_substitute = LineSubstitute(
                                old_val=old_val,
                                ent_start_char=entity_substitute.start_char-line_info.line_start_char,
                                ent_end_char=entity_substitute.end_char-line_info.line_start_char,
                                sub_spans=[],
                                entity_substitute=entity_substitute,
                                line_info=line_info
                        )
                        for span in line_info.spans:
                            if (line_substitute.ent_start_char >= span["text_start"] and 
                                line_substitute.ent_start_char < span["text_end"]) or \
                               (line_substitute.ent_end_char > span["text_start"] and 
                                line_substitute.ent_end_char <= span["text_end"]):
                                line_substitute.sub_spans.append(span)
                        line_subs.append(line_substitute)
                        lines_subs[LinePos(line_info.page_number, line_info.line_bbox)] = line_subs

    return lines_subs


def extract_pdf_images(pdf: Document) -> List[ImageInfo]:
    """
    Returns a list of all images (directly or indirectly) referenced by the page.

    Args:
        pdf: a PyMuPDF Document object

    Returns:
        A list of ImageInfo objects wrapping information about images in the page.
    """

    images = []

    for page in pdf:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(ImageInfo(
                xref=xref, 
                page_number=page.number,
                ext=base_image["ext"], 
                data=Image.open(io.BytesIO(image_bytes))
            ))

    return images


def extract_pdf_drawings(pdf: Document) -> Dict[int, List]:
    """
    Returns a list of vector graphics found in the page.

    Args:
        pdf: a PyMuPDF Document object

    Returns:
        a dictionary where key represents page number and the value a list vector graphics object.
    """

    drawings = {}

    for page in pdf:
        drawings[page.number] = page.get_drawings()

    return drawings


def save_images(images: List[ImageInfo], 
                dest_dir: str=tempfile.gettempdir()) \
                    -> Generator[SavedImageInfo, None, None]:
    """
    Saves provided images to a local file system and returns their absolute path.
    By default, images will be saved to a folder where temporary files are stored.

    Args:
        images: a list ImageInfo objects.
        dest_dir: path to a folder where images will be saved.

    Returns:
        Absolute path to the saved images.
    """

    for img in images:
        img_name = f"{img['page_number']}_{img['xref']}.{img['ext']}"
        img_path = os.path.join(dest_dir, img_name)
        img["data"].save(img_path)

        yield SavedImageInfo(
            img_path=img_path,
            xref=img["xref"],
            page_number=img["page_number"]
        )
    

def save_drawings(drawings: List, 
                  page_width: int, 
                  page_height: int, 
                  dest_dir: str=tempfile.gettempdir()) \
                  -> Generator[SavedDrawingInfo, None, None]:
    """
    Saves each drawing, to an image, on a local file system and returns 
    information about each saved image containing drawing.

    Args:
        drawings: a list of vector graphics in a page.
        page_width: initial width of the image containing drawings.
        page_height: initial height of the image containing drawings.
        dest_dir: a path to the folder where images, containing drawings will be saved.

    Returns:
        For each image containing a drawing, it returns information about the image, 
        such as the image path, its boundaries and the number of the pages where the 
        drawings were found.
    """

    for (page_num, page_drawings) in drawings.items():
        image = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(image)
        #all_drawings_bbox = None
    
        for idx, drawing in enumerate(page_drawings):
            drawing_bbox = _draw_paths(draw, drawing["items"])

            """
            if drawing_bbox:
                if not all_drawings_bbox:
                    all_drawings_bbox = drawing_bbox
                    continue
                all_drawings_bbox = (
                    min(all_drawings_bbox[0], drawing_bbox[0]),
                    min(all_drawings_bbox[1], drawing_bbox[1]),
                    max(all_drawings_bbox[2], drawing_bbox[2]),
                    max(all_drawings_bbox[3], drawing_bbox[3])
                )
            """
            #cropped_image = image.crop(all_drawings_bbox)
            if drawing_bbox:
                cropped_image = image.crop(drawing_bbox)
                if not 0 in cropped_image.size:
                    img_name = f"{page_num}_{idx}.jpeg"
                    img_path = os.path.join(dest_dir, img_name)
                    cropped_image.save(img_path)

                    yield SavedDrawingInfo(img_path, drawing_bbox, page_num)


def _draw_paths(draw: ImageDraw, paths: List) -> Optional[Tuple[float, float, float, float]]:
    """
    Draws supported path types such as line and cubic Bézier curve in an image.

    Args:
        draw: PIL ImageDraw module.
        paths: a list of tuples containing information about the object that need to be draw.
    
    Returns:
        if there is at least one supported path, in the list of paths, it returns coordinates of 
        bounding box containing all drawings, otherwise it returns None. 
    """

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = 0, 0
    bbox = [min_x, min_y, max_x, max_y]

    for path in paths:
        if path[0] == "l":  # Line segment
            _draw_line(draw, path)
            bbox = _update_bbox(min_x, min_y, max_x, max_y, *path[1:])
        #elif path[0] == "re":  # Rectangle
        #    _draw_rectangle(draw, path)
        #    bbox = _update_bbox(min_x, min_y, max_x, max_y, path[1][:2], path[1][2:])
        elif path[0] == "c":  # Curve (quadratic Bezier)
            _draw_curve(draw, path)
            bbox = _update_bezier_bbox(min_x, min_y, max_x, max_y, path[1:])

        min_x, min_y, max_x, max_y = bbox

    return None if float("inf") in bbox else bbox


def _draw_line(draw: ImageDraw, path: Tuple[str, Point, Point]) -> None:
    """
    Draws a line.

    Args:
        draw: PIL ImageDraw module.
        path: a tuple containing information about the line such as the starting and the ending point.

    Returns:
        None
    """
    (x0, y0), (x1, y1) = path[1:]
    draw.line((x0, y0, x1, y1), fill="black", width=2)
                    

def _draw_rectangle(draw: ImageDraw, path: Tuple[str, Rect, int]) -> None:
    """
    Draws a rectangle

    Args:
        draw: PIL ImageDraw module.
        path: a tuple containing information about the rectangle such as the rectangle coordinates.

    Returns:
        None
    """

    x0, y0, x1, y1 = path[1]
    draw.rectangle((x0, y0, x1, y1), outline="black")


def _draw_curve(draw: ImageDraw, path: Tuple[str, Point, Point, Point, Point]) -> None:
    """
    Draws a cubic Bézier curve

    Args:
        draw: PIL ImageDraw module.
        path: a tuple containing information about the curve such as the starting, the ending
              and the control points (p2 and p3).

    Returns:
        None
    """

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = path[1:]
    draw.line((x0, y0, x1, y1, x2, y2, x3, y3), fill="black", width=2)             


def _update_bbox(min_x: float, 
                 min_y: float, 
                 max_x: float, 
                 max_y: float, 
                 *points: Point) \
                 -> Tuple[float, float, float, float]:
    """
    Calculates and returns boundaries for a bounding box that surrounds 
    provided list of coordinates.

    Args:
        min_x: initial minimum X coordinate for the bounding box.
        min_y: initial minimum Y coordinate for the bounding box.
        max_x: initial maximum X coordinate for the bounding box.
        max_y: initial maximum Y coordinate for the bounding box.
        points: a sequence of tuple objects representing coordinates.

    Returns:
        the coordinates of the bounding box.
    """

    for (x, y) in points:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return min_x, min_y, max_x, max_y


def _update_bezier_bbox(min_x: float, 
                        min_y: float, 
                        max_x: float, 
                        max_y: float, 
                        points: List[Tuple[float, float]]) \
                 -> Tuple[float, float, float, float]:
    """
    Calculates and returns boundaries for a bounding box that surrounds 
    provided list of coordinates.

    Args:
        min_x: initial minimum X coordinate for the bounding box.
        min_y: initial minimum Y coordinate for the bounding box.
        max_x: initial maximum X coordinate for the bounding box.
        max_y: initial maximum Y coordinate for the bounding box.
        points: a sequence of tuple objects representing coordinates.

    Returns:
        the coordinates of the bounding box.
    """

    (x0, y0), (_, y1), (x2, y2), (x3, y3) = points

    return (
        min(min_x, x0),
        min(min_y, y0, y1, y2, y3),
        max(max_x, x2, x3),
        max(max_y, y0, y1, y2, y3)
    )


def _get_images_redaction(pdf: Document, 
                         images_text: Dict[str, ImageTextInfo], 
                         fill_color: Tuple[float, float, float]=(0, 0, 0)) \
                         -> List[RedactInfo]:
    """
    For each of the text lines found in images, inside a PDF document, returns 
    an information about the area that needs to be redacted.

    Args:
        pdf: PyMuPDF's Document object.
        images_text: a dictionary object where key refers to the image path on a local file system
                     and the value to the ImageTextInfo object.
        fill_color: a color that will be used to fill in the redaction rectangle.

    Returns:
        a list of RedactInfo objects.
    """

    redactions = []
    nlp = load_nlp_model()

    for img_path, img_info in images_text.items():
        page_number, img_xref = [int(i) for i in Path(img_path).stem.split("_")]
        img_rect = pdf[page_number].get_image_rects(img_xref)[0]

        for line in img_info.text_lines:
            doc = nlp(line["text"])

            if doc.ents:
                width_scale = img_rect.width / img_info.image_size[0]
                height_scale = img_rect.height / img_info.image_size[1]
                text_rect = Rect(
                    x0=img_rect[0] + line["bbox"][0] * width_scale,
                    y0=img_rect[1] + line["bbox"][1] * height_scale,
                    x1=img_rect[0] + line["bbox"][2] * width_scale,
                    y1=img_rect[1] + line["bbox"][3] * height_scale
                )
                redactions.append(
                    RedactInfo(
                        page_number=page_number,
                        rect=text_rect,
                        fill_color=fill_color
                    ))

    return redactions


def get_drawings_redaction(analysis_result: List[Tuple[str, bool]], 
                           drawings_images_info: List[SavedDrawingInfo], 
                           fill_color: float=(0, 0, 0)) \
                           -> List[RedactInfo]:
    """
    Returns metadata about drawings redaction.

    Args:
        analysis_result: prediction results from the handwrittings recognition model.
        drawings_images_info: metadata about saved images containing drawings.
        fill_color: redaction rectangle background color.

    Returns:
        a list of redaction metadata related to the images containing vector graphics.
    """

    redactions = []

    for img_path, is_handwritten_signature in analysis_result:
        if is_handwritten_signature:
            drawings_info = [d for d in drawings_images_info if d.img_path == img_path][0]
            redactions.append(
                RedactInfo(
                    page_number=drawings_info.page_number,
                    rect=drawings_info.crop_bbox,
                    fill_color=fill_color
                ))

    return redactions


def _add_redactions(pdf: Document, 
                    text_redactions: List[TextRedactInfo],
                    images_redactions: List[RedactInfo],
                    drawings_redactions: List[RedactInfo],
                    fill_color=(0, 0, 0)) -> None:
    """
    Adds redaction annotations areas for text, images and drawings in the PDF document.

    Args:
        pdf: PyMuPDF Document object.
        text_redactions: a list of text redactions.
        images_redactions: a list of images redactions.
        drawings_redactions: a list of drawings redactions.
        fill_color: the fill color of the redaction rectangle.

    Returns:
        None
    """

    rects_per_page = {}

    _ = [rects_per_page.setdefault(r["page_number"], []).append(r) 
         for redacts in [text_redactions, images_redactions, drawings_redactions]
         for r in redacts]
    
    # Merging rectangles should be done pagewise
    for page_num, rects in rects_per_page.items():
        rects_bboxes = [r["rect"] for r in rects]
        merged_rects = merge_intersecting_rects(rects_bboxes) if len(rects_bboxes) > 1 else rects_bboxes
        for rect in merged_rects:
            pdf[page_num].add_redact_annot(rect, fill=fill_color)


def _draw_redact_rects(pdf: Document, redacts: List[TextRedactInfo]) -> None:
    """
    Draws redaction rectangle.

    Args:
        pdf: PyMuPDF Document object.
        redacts: a list of texts' redactions.

    Returns:
        None
    """

    for redact in redacts:
        pdf[redact["page_number"]].draw_rect(redact["draw_rect"], fill=redact["fill_color"])


def _process_pdf_images(pdf: Document) -> List[RedactInfo]:
    """
    Extracts, saves and process all images in the PDF document and returns
    information about redacted images

    Args:
        pdf: PyMuPDF Document object.

    Returns:
        a list of RedactInfo object containing images redaction information.
    """

    images = extract_pdf_images(pdf)
    saved_images_info = list(save_images(images))
    images_text = analyze_print(saved_images_info)
    images_redactions = _get_images_redaction(pdf, images_text)
       
    return images_redactions


def _process_pdf_drawings(pdf: Document) -> List[RedactInfo]:
    """
    Extracts, saves and process all drawings in the PDF document and returns
    information about redacted drawings.

    Args:
        pdf: PyMuPDF Document object.

    Returns:
        a list of RedactInfo object containing drawings redaction information.
    """

    drawings = extract_pdf_drawings(pdf)
    first_page = pdf[0]
    drawings_images_info = list(save_drawings(drawings, int(first_page.rect.width), int(first_page.rect.height)))
    drawing_images_paths = [d.img_path for d in drawings_images_info]
    results = analyze_handwriting(drawing_images_paths)
    drawings_redactions = get_drawings_redaction(results, drawings_images_info)

    return drawings_redactions


def _process_text_content(pdf: Document) -> Tuple[List[Paragraph], List[TextRedactInfo]]:
    """
    Returns the original text paragraphs along with the list of all text redactions
    in those paragraphs.

    Args:
        pdf: PyMuPDF Document object.

    Returns:
        a tuple with a list of all paragraphs and a list of all text redactions in those paragraphs.
    """

    paragraphs = get_paragraphs(pdf)
    lines_subs = run_nlp(paragraphs)

    return paragraphs, substitute_page_entities(pdf, lines_subs)


def _apply_redactions(
    pdf: Document,
    text_redactions: List[TextRedactInfo],
    images_redactions: List[RedactInfo],
    drawings_redactions: List[RedactInfo]
) -> None:
    """
    Applies redactions of PDF objects and draws redaction rectangles.

    Args:
        text_redactions: a list of text redactions.
        images_redactions: a list of images redactions.
        drawings_redactions: a list of drawings redactions.

    Returns:
        None
    """

    _add_redactions(pdf, text_redactions, images_redactions, drawings_redactions)   

    for page in pdf:
        page.apply_redactions() 

    _draw_redact_rects(pdf, text_redactions)
    

def _save_processed_document(pdf: Document, output_path: str) -> None:
    """
    Stores the optimized version of the document, regarding its size, to the provided path.

    Args:
        pdf: PyMuPDF Document object.
        output_path: a path where the PDF document should be saved.
    """

    pdf.subset_fonts()
    pdf.ez_save(output_path.absolute().as_posix())


def _get_processing_document_paths(input_file: str, output_dir: str) -> Tuple[Path, Path]:
    """
    For the given path of the input PDF document, returns the path to processed PDF 
    document in the output folder.

    Args:
        input_file: a path to the input document.
        output_dir: a path to the folder where the processed document should be stored.

    Returns:
        a tuple consisting of the Path object for the input and the output files.
    """

    input_file_path = Path(input_file)
    output_path = Path(output_dir).joinpath(input_file_path.name)
    os.makedirs(output_dir, exist_ok=True)
    
    return input_file_path, output_path


def process_document(input_file: str, output_dir: str, reconstruct: bool) -> None:
    """
    The main entry point of the script.

    Args:
        input_file: a file path to a PDF document.
        output_dir: a path to a folder where the final PDF document will be stored.
        reconstruct: whether or not to reconstruct the text in paragraph, used to
                     readd the original text in the lines surrounding the line that's
                     been redacted.

    Returns:
        None
    """
    
    input_file_path, output_path = _get_processing_document_paths(input_file, output_dir)
     
    with fitz.open(input_file_path.absolute().as_posix()) as pdf:
        images_redactions = _process_pdf_images(pdf)
        drawings_redactions = _process_pdf_drawings(pdf)
        paragraphs, text_redactions = _process_text_content(pdf)
        
        if reconstruct:
            _reconstruct_deleted_text(pdf, text_redactions, paragraphs)
    
        _apply_redactions(
            pdf,
            text_redactions,
            images_redactions,
            drawings_redactions
        )

        _save_processed_document(pdf, output_path)
