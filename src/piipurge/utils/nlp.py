import re
import spacy
import string
import random
import logging
import unicodedata
from .. import consts
from re import Pattern
from typing import List
from torch import Tensor
from ..schemas import Paragraph
from itertools import filterfalse
from spacy.tokens import Doc, Span
from difflib import SequenceMatcher
from spacy.language import Language
from spacy.util import filter_spans
from collections import defaultdict
from spacy.pipeline import Sentencizer
from .synth_text_generator import generate_text
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict, Optional, Union, Callable


logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)


def generate_examples(
    doc_annots: Tuple[str, List[Tuple[int, int, str]]], max_examples: int
) -> List[Tuple[str, List]]:
    return generate_text(doc_annots, max_examples)


class EntityBalancer:

    def __init__(self, ngram_delim=" "):
        self.ngram_delim = ngram_delim


def is_single_entity_annotation(annot: Tuple[str, list]) -> bool:
    """
    Checks whether the the list of annotations has just a single annotation.

    Args:
        annot: a tuple consisting of a text and a list of annotations related to the text

    Returns:
        True if there is a single annotations in the list of annotations otherwise False.
    """
    # annotations are in format (SENTENCE_TEXT, [(START_CHAR, END_CHAR, LABEL1),(START_CHAR, END_CHAR, LABEL2),...,])
    # a[2] refers to the annotation label e.g. ORG, PERSON etc.
    entity_types = set([a[2] for a in annot[1]])

    return len(entity_types) == 1


def _group_entity_examples(
    docs_annots: List[Tuple[str, List]],
) -> Dict[str, List[Tuple[str, List]]]:
    """
    Group annotations by the entity type.

    Args:
        docs_annots: a list of all annotations in a document where each tuple consists 
                     of a sentence and the list of annotations related to that sentence.

    Returns:
        a dictionary where keys are entity types or labels and values a list of annotations
        related to the particular entity type.
    """

    annots_counts = defaultdict(list)

    for text, annotations in docs_annots:
        entity_types = set([a[2] for a in annotations])
        for ent_type in entity_types:
            annots_counts[ent_type].append((text, annotations))

    return dict(annots_counts)


def _oversample(
    annots_groupped_by_entity: Dict[str, List[Tuple[str, List]]],
    max_majority_minority_ratio: float,
) -> Dict[str, List[Tuple[str, List]]]:
    """
    Oversamples the annotations belonging to the minority entity classes.

    Args:
        annots_groupped_by_entity: annotations groupped by the entity type.
        max_majority_minority_ratio: a threshold used when deciding whether or not to oversample
                                     the annotations for the particular entity type, any value
                                     greater then the ratio would indicate that the number of 
                                     samples related to a particular entity class is lower than
                                     required.

    Returns:
        a balanced annotations dictionary groupped by entity type.
    """

    if annots_groupped_by_entity is None or len(annots_groupped_by_entity) == 0:
        return {}
    
    sorted_annots = dict(
        sorted(annots_groupped_by_entity.items(), key=lambda i: len(i[1]), reverse=True)
    )

    max_count = max([len(annots) for _, annots in sorted_annots.items()])
    entity_types = list(sorted_annots.keys())
    new_single_entities_annots = {entity_types[0]: sorted_annots[entity_types[0]]}

    for entity_type in entity_types[1:]:
        entity_type_examples_len = len(sorted_annots[entity_type])
        examples_ratio = max_count / entity_type_examples_len
        if examples_ratio > max_majority_minority_ratio:
            total_examples_to_generate = int(
                round(entity_type_examples_len * examples_ratio)
            )
            num_augmentations_per_example = (
                total_examples_to_generate // entity_type_examples_len
            ) + 1
            augmented_examples = generate_text(
                sorted_annots[entity_type], num_augmentations_per_example
            )
            new_single_entities_annots[entity_type] = augmented_examples
        else:
            new_single_entities_annots[entity_type] = sorted_annots[entity_type]

    return new_single_entities_annots


def _undersample(
    annots_groupped_by_entity: Dict[str, List[Tuple[str, List]]],
    max_majority_minority_ratio: float,
) -> Dict[str, List[Tuple[str, List]]]:
    """
    Undersamples the annotations related to majority entity classes.

    Args:
        annots_groupped_by_entity: annotations groupped by the entity type.
        max_majority_minority_ratio: a threshold used when deciding whether or not to undersample
                                     the annotations for the particular entity type, any value
                                     greater then the ratio would indicate that the number of 
                                     samples related to a particular entity class is greater than
                                     required.

    Returns:
        a balanced annotations dictionary groupped by entity type.
    """

    if annots_groupped_by_entity is None or len(annots_groupped_by_entity) == 0:
        return {}  
    sorted_annots = dict(
        sorted(
            annots_groupped_by_entity.items(), key=lambda i: len(i[1])
        )
    )
    min_count = min([len(annots) for _, annots in sorted_annots.items()])
    entity_types = list(sorted_annots.keys())
    new_single_entities_annots = {entity_types[0]: sorted_annots[entity_types[0]]}
    # Maximum allowed number of samples per entity type.
    max_allowed_samples = int(round(max_majority_minority_ratio) * min_count)

    for entity_type in entity_types[1:]:
        entity_type_examples_len = len(sorted_annots[entity_type])
        examples_ratio = entity_type_examples_len / min_count
        if examples_ratio > max_majority_minority_ratio:
            new_single_entities_annots[entity_type] = random.sample(
                sorted_annots[entity_type], max_allowed_samples
            )
        else:
            new_single_entities_annots[entity_type] = sorted_annots[entity_type]

    return new_single_entities_annots


def balance_examples(
    docs_annots: List[Tuple[str, List]], 
    max_majority_minority_ratio: float = 1.5
) -> List[Tuple[str, List]]:
    """
    Balances annotations by oversampling annotations belonging the minority and
    undersampling annotations belonging to the majority class. 

    Args:
        annots_groupped_by_entity: annotations groupped by the entity type.
        max_majority_minority_ratio: a threshold used when deciding whether or not to undersample
                                     or oversample the annotations for the particular entity type, 
                                     any value greater then the ratio would indicate that the number 
                                     of samples related to a particular entity class is greater or lower
                                     than required.
    """

    single_entity_annots = list(filter(is_single_entity_annotation, docs_annots))
    grouped_single_entity_annots = _group_entity_examples(single_entity_annots)
    multiple_entities_annots = list(
        filterfalse(is_single_entity_annotation, docs_annots)
    )
    grouped_single_entity_annots = dict(
        sorted(grouped_single_entity_annots.items(), key=lambda i: len(i[1]))
    )
    augmented_single_entities_annots = _oversample(
        grouped_single_entity_annots, max_majority_minority_ratio
    )
    single_entities_annots = _undersample(
        augmented_single_entities_annots, max_majority_minority_ratio
    )

    balanced_examples = [j for i in single_entities_annots.values() for j in i] + \
                        [i for i in multiple_entities_annots]

    return balanced_examples


def _find_matching_entities(doc: Doc, regex: Pattern, label: str) -> Doc:
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
    spans = [
        doc.char_span(match.start(), match.end(), label=label) for match in matches
    ]
    spans = [s for s in spans if s is not None]
    spans = filter_spans(list(doc.ents) + spans)
    doc.ents = spans

    return doc


@Language.component("ipv4")
def ipv4_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["ipv4"], "IPv4")

    return doc


@Language.component("ipv6")
def ipv6_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["ipv6"], "IPv6")

    return doc


@Language.component("phone")
def phone_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["phone"], "PHONE")

    return doc


@Language.component("email")
def email_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["email"], "EMAIL")

    return doc


@Language.component("ssn")
def ssn_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["ssn"], "SSN")

    return doc


@Language.component("medicare")
def medicare_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["medicare"], "MEDICARE")

    return doc


@Language.component("vin")
def vin_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["vin"], "VIN")

    return doc


@Language.component("url")
def url_component(doc: Doc):
    doc = _find_matching_entities(doc, consts.PATTERNS["url"], "URL")

    return doc


@Language.component("custom_sentencizer")
def get_sentencizer(doc: Doc):
    sentencizer = Sentencizer(punct_chars=[r"\n"])

    return sentencizer(doc)


def _normalize_utf8_text(text):
    """ """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Replace no-break space with regular space
    text = text.replace("\u00a0", " ")

    # Translate some special punctuation characters
    translation_table = {
        ord("\u2018"): "'",
        ord("\u2019"): "'",
        ord("\u201c"): '"',
        ord("\u201d"): '"',
        ord("\u2013"): "-",
        ord("\u2014"): "-",
        ord("\u2026"): "...",
        ord("\u00b7"): "-",
    }
    text = text.translate(translation_table)

    # Remove zero-width spaces and formatting chars
    text = re.sub(r"[\u200B\u200E\u200F\uFEFF]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_link(text: str) -> bool:
    """
    Check whether a string refers to an URL or an e-mail address.

    Args:
        text: text to check.

    Returns:
        True if a string represents an URL or e-mail address otherwise False.
    """

    return (
        consts.PATTERNS["url"].match(text) is not None
        or consts.PATTERNS["email"].match(text) is not None
    )


def get_spans_similarity(
    span1: str, span2: str, text_encoder: SentenceTransformer
) -> Tensor:
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


def check_common_acronyms(
    span_text: str,
    common_acronyms: list[str],
    text_encoder: SentenceTransformer,
    similarity_score_threshold: float = 0.95,
) -> str | None:
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
        if (span_similarity_score > max_similarity) and (
            span_similarity_score > similarity_score_threshold
        ):
            optim_acronym = acronym
            max_similarity = span_similarity_score

    return optim_acronym


def get_most_similar_text(
    texts_to_compare: Optional[List[str]],
    texts: Optional[List[Union[str, Tuple[str, str]]]],
    text_encoder: SentenceTransformer,
    similarity_score_threshold: float = 0.95,
) -> Tuple[str, str] | None:
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
        A tuple from the `texts` list, consisting of the acronym and its expanded form, 
        that is most similar to any of the `texts_to_compare` and has a similarity score 
        above the threshold. Returns None if no match meets the threshold.
    """

    most_similar_text = None

    if (texts_to_compare is not None) and (texts is not None):
        curr_max_similarity = 0
        for text1 in texts_to_compare:
            for text2 in texts:
                if isinstance(text2, tuple):
                    text_similarity_score = max(
                        get_spans_similarity(text2[0], text1, text_encoder),
                        get_spans_similarity(text2[1], text1, text_encoder),
                    )
                else:
                    text_similarity_score = get_spans_similarity(
                        text2, text1, text_encoder
                    )
                if (text_similarity_score > curr_max_similarity) and (
                    text_similarity_score > similarity_score_threshold
                ):
                    most_similar_text = text2
                    curr_max_similarity = text_similarity_score

    return most_similar_text


def add_utf8_normalization(nlp: Language):
    original_make_doc = nlp.make_doc

    def make_doc_with_utf8_normalization(text):
        normalized_text = _normalize_utf8_text(text)
        return original_make_doc(normalized_text)

    nlp.make_doc = make_doc_with_utf8_normalization


def load_nlp_model(model_name: str = "en_core_web_trf") -> Language:
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
    patterns = [
        {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]}
    ]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer")
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
    nlp_acronyms.add_pipe("sentencizer")

    return nlp_acronyms


def load_text_encoder_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    encoder_model = SentenceTransformer(model_name)

    return encoder_model


def get_ent_replacement(
    ent: Span, suffix_type: str | int, entities_count: int = 0
) -> str:
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
        suffix = (
            int(round(entities_count / len(string.ascii_uppercase))) * "A"
            + string.ascii_uppercase[entities_count % len(string.ascii_uppercase)]
        )
        return f'"{consts.ENTITY_DESC[ent.label_][0]} {suffix}"'
    elif suffix_type is int:
        return f'"{consts.ENTITY_DESC[ent.label_][0]} {entities_count+1}"'

    raise Exception(f"Unsupported suffix type '{suffix_type}")


def _find_best_matching_string(
    source_text: str,
    target_text: str,
    start_char: int,
    end_char: int,
    best_match_ratio: float,
    iterate_fn: Callable[[int, int], Tuple[int, int]],
) -> Tuple[float, int, int]:
    """
    Finds the best matching substring in the target text that matches the source text
    by iteratively expanding or contracting the substring boundaries.

    Args:
        source_text: the text to match against.
        target_text: the text in which to search for the best match.
        start_char: starting character index in the target text.
        end_char: ending character index in the target text.
        best_match_ratio: initial best match ratio.
        iterate_fn: a function that defines how to iterate over the substring boundaries.

    Returns:
        A tuple containing the best match ratio, the starting character index, and the ending character index
        of the best matching substring in the target text.
    """
    if start_char < 0 or end_char > len(target_text):
        raise ValueError(
            f"start_char {start_char} and end_char {end_char} must be within the bounds of target_text length {len(target_text)}."
        )
    
    if start_char > end_char:
        raise ValueError(
            f"start_char {start_char} must not be greater than end_char {end_char}."
        )
    
    if not source_text or not target_text or \
       len(source_text) == 0 or len(target_text) == 0:
        return 0.0, start_char, end_char

    sm = SequenceMatcher(None, source_text, target_text[start_char:end_char])
    old_start_char, old_end_char = start_char, end_char
    new_best_match_ratio = best_match_ratio
    while True:
        match_ratio = sm.ratio()
        if match_ratio < new_best_match_ratio or start_char < 0 or \
           end_char > len(target_text):
            return new_best_match_ratio, old_start_char, old_end_char
        elif match_ratio == 1.0:
            return match_ratio, start_char, end_char
        new_best_match_ratio = match_ratio
        old_start_char, old_end_char = start_char, end_char
        start_char, end_char = iterate_fn(start_char, end_char)
        sm.set_seq2(target_text[start_char:end_char])


def find_entity_boundaries(ent: Span, paragraph: Paragraph) -> Tuple[int, int, float]:
    text_index = paragraph["text"].find(ent.text)
    if text_index != -1:
        return text_index, text_index + len(ent.text), 1.0
    char_start_index = _normalize_utf8_text(paragraph["text"]).index(ent.text)
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        ent.text,
        paragraph["text"],
        char_start_index,
        char_start_index + len(ent.text),
        0.0,
        lambda s, e: (s + 1, e + 1),
    )
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        paragraph["text"][start_char:end_char],
        paragraph["text"],
        start_char,
        start_char + len(ent.text),
        best_match_ratio,
        lambda s, e: (s, e + 1),
    )
    best_match_ratio, start_char, end_char = _find_best_matching_string(
        paragraph["text"][start_char:end_char],
        paragraph["text"],
        start_char,
        end_char,
        best_match_ratio,
        lambda s, e: (s - 1, e),
    )
    logging.info(f"(_find_entity_boundaries): best_match_ratio: {best_match_ratio}")

    return start_char, end_char, best_match_ratio
