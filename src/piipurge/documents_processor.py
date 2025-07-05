import os
import re
import fitz
import logging
import tempfile
import datasets
from . import consts
from pathlib import Path
from spacy.tokens import Doc, Span
from spacy.language import Language
from .utils.drawing import save_drawings
from .utils.fonts import get_text_length
from .utils.pdf import (
    replace_text, fill_bg, draw_paragraph_annots, substitute_page_entities,
    apply_redactions, get_image_rects, draw_redact_rects, extract_pdf_images,
    get_paragraphs, extract_pdf_drawings, save_processed_document
)
from .utils.nlp import (
    load_nlp_model, load_nlp_acronyms_model, load_text_encoder_model,
    check_common_acronyms, get_most_similar_text, get_ent_replacement, 
    find_entity_boundaries
)
from pymupdf import Document, Page, Rect, Pixmap
from .analyze_images import analyze_print, analyze_handwriting
from .schemas import (
    LinePos,
    EntitySubstitute,
    SpanInfo,
    LineInfo,
    LineSubstitute,
    SavedDrawingInfo,
    SavedImageInfo,
    Paragraph,
    ImageInfo,
    RedactInfo,
    TextRedactInfo,
    ImageTextInfo,
)
from typing import (
    Set,
    List,
    Tuple,
    Optional,
    Dict,
    Sequence,
    Generator
)

datasets.disable_progress_bar()
logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)



def get_closest_ent_name(
    texts: List[str],
    ent_cat_subs: Set[str],
    relations: List[Tuple[str, str]],
    text_encoder: object,
) -> Optional[str]:
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
        closest_ent_name = (
            relation[0]
            if relation[0] in ent_cat_subs
            else relation[1] if relation[1] in ent_cat_subs else None
        )
        if closest_ent_name is not None:
            closest_ent_name = get_most_similar_text(texts, ent_cat_subs, text_encoder)

    return closest_ent_name


def _update_subs(
    subs: Dict[str, Dict[str, EntitySubstitute]],
    ents: List[Span],
    relations: List[Tuple[str, str]],
    common_acronyms: List[str],
    text_encoder: object,
    paragraph: Paragraph,
) -> None:
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
        text_encoder: the model used for text encoding.
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
            _process_org_entity(
                ent, relations, ent_subs, text_encoder, suffix_type, paragraph
            )
        else:
            _process_standard_entity(ent, ent_subs, suffix_type, paragraph)

        subs[ent.label_] = ent_subs


def _should_process_entity(
    ent: Span, common_acronyms: List[str], text_encoder: object
):
    """
    Determines whether the text representing a named entity should be further processed.

    Args:
        ent: named entity object
        common_acronyms: a list of strings representing acronyms
        text_encoder: the model used for the encoding

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


def _process_org_entity(
    ent: Span,
    relations: List[Tuple[str, str]],
    ent_subs: dict[str, List[EntitySubstitute]],
    text_encoder: object,
    suffix_type: str,
    paragraph: Paragraph,
) -> None:
    """
    Determines replacement string for the ORG named entity.

    Args:
        ent: named entity object
        relations: a list, where each element represents a combination of an acronym and its longer form.
        ent_subs: a dictionary of all string values, related to the recognized named entities, in a document
                  and their replacement strings where the tuple value is in the following format:
                  (REPLACEMENT_STRING, STARTING_CHAR_IN_DOCUMENT, ENDING_CHAR_IN_DOCUMENT,
                   STARTING_CHAR_IN_SENTENCE, ENDING_CHAR_IN_SENTENCE, PARAGRAPH_INDEX)
        text_encoder: the model used for text encoding
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
    closest_key_name = get_closest_ent_name(
        texts_to_compare, ent_subs, temp_relations, text_encoder
    )

    if closest_key_name:
        ent_replacements = ent_subs[closest_key_name]
        _ = [
            ent_replacements.append(s)
            for s in _create_entity_substitute(
                ent, ent_subs[closest_key_name][0][0], paragraph
            )
        ]
        return

    ent_subs[ent.text] = list(
        _create_entity_substitute(
            ent, get_ent_replacement(ent, suffix_type, len(ent_subs)), paragraph
        )
    )


def _process_standard_entity(
    ent: Span,
    ent_subs: dict[str, List[EntitySubstitute]],
    suffix_type: str,
    paragraph: Paragraph,
) -> None:
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
        _ = [
            replacements.append(s)
            for s in _create_entity_substitute(ent, ent_subs[ent.text][0][0], paragraph)
        ]
    else:
        ent_subs[ent.text] = list(
            _create_entity_substitute(
                ent, get_ent_replacement(ent, suffix_type, len(ent_subs)), paragraph
            )
        )


def _is_entity_inside_line(ent_start_char: int, ent_end_char) -> bool:
    def fn(line):
        return (
            ent_start_char >= line.line_start_char
            and ent_start_char <= line.line_start_char
        ) or (
            ent_end_char >= line.line_start_char and ent_end_char <= line.line_end_char
        )

    return fn


def _create_entity_substitute(
    ent: Span, new_val: str, paragraph: Paragraph, min_match_ratio=0.95
) -> Generator[EntitySubstitute, None, None]:
    ent_start_char, ent_end_char, best_match_ratio = find_entity_boundaries(
        ent, paragraph
    )
    if best_match_ratio < min_match_ratio:
        raise f"Text {ent.text} not found in the paragraph {paragraph['text']}"
    lines = list(
        filter(_is_entity_inside_line(ent_start_char, ent_end_char), paragraph["lines"])
    )

    return (
        EntitySubstitute(
            old_val=paragraph["text"][ent_start_char:ent_end_char],
            new_val=new_val,
            start_char=ent_start_char,
            end_char=ent_start_char + len(ent.text),
            sent_start_char=max(
                ent_start_char - line.line_start_char, line.line_start_char
            ),
            sent_end_char=min(ent_end_char - line.line_start_char, line.line_end_char),
            line_index=line.line_index,
            paragraph_index=paragraph["index"],
            ent_type=ent.label_,
        )
        for line in lines
    )


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

    long_acronym = "".join(ent.text[0].upper() for ent in long_form_entities)
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
            long_form = [ent]
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

    return (
        tag.startswith("B-long") or tag.startswith("I-long")
    ) and start - long_form_range[1] > 2


def _is_continuation_of_long_form(
    tag: str, start: int, long_form_range: Tuple[int, int]
) -> bool:
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

    return (tag.startswith("I-long")) or (
        tag.startswith("B-long") and (start - long_form_range[1] <= 2)
    )


def _is_short_form(
    tag: str,
    start: int,
    long_form: List[Span],
    long_form_range: Tuple[int, int],
    ent: Span,
) -> bool:
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

    if (
        not tag.startswith("B-short")
        or not long_form
        or (start - long_form_range[1]) > 2
    ):
        return False

    return is_acronym(ent.text, long_form) or (
        start - long_form_range[1] <= 2 and long_form[-1].sent == ent.sent
    )


def get_redacted_text(line_subs: List[LineSubstitute], redact_char="x") -> str:
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
    sorted_line_subs = list(
        sorted(line_subs, key=lambda i: i.ent_start_char, reverse=True)
    )

    for sub in sorted_line_subs:
        redacted_text = redact_text(sub, redacted_text, redact_char)

    return redacted_text


def redact_text(sub: LineSubstitute, redacted_text: str, redact_char: str) -> str:
    """ 
    Performs text redaction by replacing the text with the specified redaction character.

    Args:
        sub: contains information about the text value that need to be replaced.
        redacted_text: a line of text that's being redacted.
        redact_char: a replacement character used to replace each character in the redacted text.

    Returns:
        a redacted line of text.
    """

    # It's assumed that all spans in a line have the same font name and the font size.
    # For more robust solution each span's font name and font size should be inspected
    # separately.
    font = sub.line_info.spans[0]["font"]
    size = sub.line_info.spans[0]["size"]

    new_val = len(sub.old_val) * redact_char
    while get_text_length(new_val, font_name=font, font_size=size) >= get_text_length(
        sub.old_val, font_name=font, font_size=size
    ):
        new_val = new_val[:-1]
    text = (
        redacted_text[:sub.ent_start_char]
        + new_val
        + redacted_text[sub.ent_end_char:]
    )

    return text


def is_neighboring_line(line: LineInfo, lines: List[int]) -> bool:
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


def get_redacted_paragraphs(
    annotations: List[TextRedactInfo],
) -> Dict[Tuple[int, int], List[Tuple[Rect, LineSubstitute]]]:
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
        key = (
            line_sub.line_info.page_number,
            line_sub.entity_substitute.paragraph_index,
        )
        parag_replacements = redacted_paragraphs.get(key, [])
        parag_replacements.append((redact_rect, line_sub))
        redacted_paragraphs[key] = parag_replacements

    return redacted_paragraphs


def _reconstruct_deleted_text(
    pdf: Document,
    redactions: List[TextRedactInfo],
    paragraphs: List[Paragraph],
    draw_redact_annot: bool = False,
) -> None:
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
        redacted_lines = [
            line_substitute.line_info.paragraph_line_index
            for (_, line_substitute) in parag_replacements
        ]

        for line_info in paragraphs[paragraph_index]["lines"]:
            if line_info.paragraph_line_index in redacted_lines or is_neighboring_line(
                line_info, redacted_lines
            ):
                if line_info.paragraph_line_index in redacted_lines:
                    text = get_redacted_text(
                        [
                            line_substitute
                            for (_, line_substitute) in parag_replacements
                            if line_substitute.line_info.paragraph_line_index
                            == line_info.paragraph_line_index
                        ]
                    )
                else:
                    text = line_info.text

                fill_bg(pdf[page_num], line_info, margin=0.2)
                replace_text(pdf[page_num], line_info, text)
                if draw_redact_annot:
                    draw_paragraph_annots(
                        pdf[page_num], line_info, redacted_lines, parag_replacements
                    )


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
    encoder_model = load_text_encoder_model()

    subs = {}
    for paragraph in paragraphs:
        # Spacy
        doc = nlp(paragraph["text"])
        acronyms_doc = nlp_acronyms(paragraph["text"])

        relations = _find_short_long_relations(acronyms_doc.ents)

        for ent in doc.ents:
            logging.debug(f"ent.text: {ent.text}, ent.label_: {ent.label_}")

        _update_subs(
            subs, doc.ents, relations, consts.COMMON_ACRONYMS, encoder_model, paragraph
        )

    lines_subs = _map_subs_to_lines(subs, paragraphs)

    return lines_subs


def _map_subs_to_lines(
    subs: Dict[str, Dict[str, List[EntitySubstitute]]], paragraphs: List[Paragraph]
) -> Dict[LinePos, List[Dict]]:
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
                    if (entity_substitute.start_char >= line_info.line_start_char) and (
                        entity_substitute.end_char <= line_info.line_end_char
                    ):
                        line_subs = lines_subs.get(
                            (line_info.page_number, line_info.line_bbox), []
                        )
                        line_substitute = LineSubstitute(
                            old_val=old_val,
                            ent_start_char=entity_substitute.start_char
                            - line_info.line_start_char,
                            ent_end_char=entity_substitute.end_char
                            - line_info.line_start_char,
                            sub_spans=[],
                            entity_substitute=entity_substitute,
                            line_info=line_info,
                        )
                        for span in line_info.spans:
                            if (
                                line_substitute.ent_start_char >= span["text_start"]
                                and line_substitute.ent_start_char < span["text_end"]
                            ) or (
                                line_substitute.ent_end_char > span["text_start"]
                                and line_substitute.ent_end_char <= span["text_end"]
                            ):
                                line_substitute.sub_spans.append(span)
                        line_subs.append(line_substitute)
                        lines_subs[
                            LinePos(line_info.page_number, line_info.line_bbox)
                        ] = line_subs

    return lines_subs


def save_images(
    images: List[ImageInfo], dest_dir: str = tempfile.gettempdir()
) -> Generator[SavedImageInfo, None, None]:
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
            img_path=img_path, xref=img["xref"], page_number=img["page_number"]
        )


def _get_images_redaction(
    pdf: Document,
    images_text: Dict[str, ImageTextInfo],
    fill_color: Tuple[float, float, float] = (0, 0, 0),
) -> List[RedactInfo]:
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
        img_rect = get_image_rects(pdf, page_number, img_xref)[0]

        for line in img_info.text_lines:
            doc = nlp(line["text"])

            if doc.ents:
                width_scale = img_rect.width / img_info.image_size[0]
                height_scale = img_rect.height / img_info.image_size[1]
                text_rect = Rect(
                    x0=img_rect[0] + line["bbox"][0] * width_scale,
                    y0=img_rect[1] + line["bbox"][1] * height_scale,
                    x1=img_rect[0] + line["bbox"][2] * width_scale,
                    y1=img_rect[1] + line["bbox"][3] * height_scale,
                )
                redactions.append(
                    RedactInfo(
                        page_number=page_number, rect=text_rect, fill_color=fill_color
                    )
                )

    return redactions


def get_drawings_redaction(
    analysis_result: List[Tuple[str, bool]],
    drawings_images_info: List[SavedDrawingInfo],
    fill_color: float = (0, 0, 0),
) -> List[RedactInfo]:
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
                    fill_color=fill_color,
                )
            )

    return redactions


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
    drawings_images_info = list(
        save_drawings(drawings, int(first_page.rect.width), int(first_page.rect.height))
    )
    drawing_images_paths = [d.img_path for d in drawings_images_info]
    results = analyze_handwriting(drawing_images_paths)
    drawings_redactions = get_drawings_redaction(results, drawings_images_info)

    return drawings_redactions


def _process_text_content(
    pdf: Document,
) -> Tuple[List[Paragraph], List[TextRedactInfo]]:
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


def _get_processing_document_paths(
    input_file: str, output_dir: str
) -> Tuple[Path, Path]:
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

    input_file_path, output_path = _get_processing_document_paths(
        input_file, output_dir
    )

    with fitz.open(input_file_path.absolute().as_posix()) as pdf:
        images_redactions = _process_pdf_images(pdf)
        drawings_redactions = _process_pdf_drawings(pdf)
        paragraphs, text_redactions = _process_text_content(pdf)

        if reconstruct:
            _reconstruct_deleted_text(pdf, text_redactions, paragraphs)

        apply_redactions(pdf, text_redactions, images_redactions, drawings_redactions)

        save_processed_document(pdf, output_path)
