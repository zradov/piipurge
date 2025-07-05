import io
import fitz 
from PIL import Image
from typing import List, Tuple, Dict
from .drawing import merge_intersecting_rects
from .fonts import get_text_length, get_matching_font
from pymupdf import Document, Page, Rect, Pixmap, Point
from ..schemas import (
    LinePos,
    SpanInfo,
    LineInfo,
    LineSubstitute,
    Paragraph,
    ImageInfo,
    RedactInfo,
    TextRedactInfo
)

fitz.TOOLS.set_small_glyph_heights(True)


def replace_text(page: Page, line: LineInfo, text: str) -> None:
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
        text_color = (
            span["color"]
            if isinstance(span["color"], tuple)
            else [round(c / 255.0, 2) for c in fitz.sRGB_to_rgb(span["color"])]
        )
        kwargs = dict(
            fontsize=span["size"], color=text_color, fill_opacity=span["alpha"]
        )

        font_metadata = get_matching_font(span["font"])
        kwargs["fontfile"] = font_metadata["font_path"]
        span_text = text[span["text_start"] : span["text_end"]]
        _ = page.insert_text(
            (span["origin"][0] + offset, span["origin"][1]), span_text, **kwargs
        )
        new_text_length_pts = get_text_length(
            span_text, font_name=font_metadata["full_name"], font_size=span["size"]
        )
        offset += new_text_length_pts - (span["span_bbox"][2] - span["span_bbox"][0])


def _get_redact_rect(
    rect: Rect, offset_y: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    Returns the exact boundaries of the redaction rectangle.

    Args:
        rect: an input rectangle.
        offset_y: offset from the center of the input rectangle at which the y-coordinates 
                  of the redaction rectangle will be set.

    Returns:
        a tuple containing the coordinates of the top-left and the bottom-right corners
        of the redaction rectangle.
    """

    center_y = rect[1] + (rect[3] - rect[1]) / 2
    redact_rect = (rect[0], center_y - offset_y, rect[2], center_y + offset_y)

    return redact_rect


def get_text_redact_info(
    page_number: int,
    rect: Rect,
    line_sub: LineSubstitute,
    use_span_bg: bool,
    default_redact_color: Tuple[float, float, float],
) -> TextRedactInfo:
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
    draw_redact_rect = [
        rect[0],
        line_sub.line_info.spans_bbox[1],
        rect[2],
        line_sub.line_info.spans_bbox[3],
    ]
    # The height of the redaction rectangle will be reduced as much as possible, in order to
    # avoid the issue with redactions of text in the neighboring lines, as describe in the issue
    # https://github.com/pymupdf/PyMuPDF/discussions/1810
    redact_rect = _get_redact_rect(rect)
    # If span background should be used as the fill color, for the redact annotation rectangle,
    # choose the fill color of the first span containing text that need to be redacted.
    redact_color = (
        line_sub.sub_spans[0]["fill_color"] if use_span_bg else default_redact_color
    )

    return TextRedactInfo(
        page_number=page_number,
        rect=redact_rect,
        draw_rect=draw_redact_rect,
        line_sub=line_sub,
        fill_color=redact_color,
    )


def fill_bg(page: Page, line: LineInfo, margin: int = 0) -> None:
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
        redact_bbox = [
            bbox[0] - margin,
            bbox[1] - margin,
            bbox[2] + margin,
            bbox[3] + margin,
        ]
        page.draw_rect(redact_bbox, color=span["fill_color"], fill=span["fill_color"])


def _get_redact_annots(
    page: Page,
    line_subs: LineSubstitute,
    use_span_bg: bool = False,
    fill_color: Tuple[float, float, float] = (0, 0, 0),
) -> List[TextRedactInfo]:
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
            rects = page.search_for(
                line_sub.old_val,
                clip=line_sub.line_info.spans_bbox,
                quads=False,
                flags=0,
            )
            for r in rects:
                redact_info = get_text_redact_info(
                    page.number, r, line_sub, use_span_bg, fill_color
                )
                redactions.append(redact_info)

    return redactions


def draw_paragraph_annots(
    page: Page,
    line: LineInfo,
    redacted_lines: List[int],
    parag_replacements: List[Tuple[Tuple[float, float, float, float], LineSubstitute]],
    redact_rect_color: Tuple[float, float, float] = (0, 0, 0),
) -> None:
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
        line_subs = [
            r
            for r in parag_replacements
            if r[1]["paragraph_line_index"] == line["paragraph_line_index"]
        ]
        for redact_rect, _ in line_subs:
            page.draw_rect(redact_rect, color=redact_rect_color, fill=redact_rect_color)


def substitute_page_entities(
    pdf: Document, lines_subs: Dict[LinePos, List[Dict]], min_distance=3
) -> List[TextRedactInfo]:
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
                        if (
                            line_subs.ent_start_char - last_org.ent_end_char
                            <= min_distance
                        ) and (
                            last_org.entity_substitute.new_val
                            == line_subs.entity_substitute.new_val
                        ):
                            # The new ORG will not be added, instead the previous ORG's ending character index
                            # will be expanded so that it includes the new ORG.
                            new_org_end_char = line_subs.ent_end_char
                            if (
                                len(last_org.line_info.text) - 1 < new_org_end_char
                            ) and (
                                last_org.line_info.text[new_org_end_char + 1] == ")"
                            ):
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
            annots = _get_redact_annots(page, replacements)
            if annots:
                redactions.extend(annots)

    return redactions


def _add_redactions(
    pdf: Document,
    text_redactions: List[TextRedactInfo],
    images_redactions: List[RedactInfo],
    drawings_redactions: List[RedactInfo],
    fill_color=(0, 0, 0),
) -> None:
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

    _ = [
        rects_per_page.setdefault(r["page_number"], []).append(r)
        for redacts in [text_redactions, images_redactions, drawings_redactions]
        for r in redacts
    ]

    # Merging rectangles should be done pagewise
    for page_num, rects in rects_per_page.items():
        rects_bboxes = [r["rect"] for r in rects]
        merged_rects = (
            merge_intersecting_rects(rects_bboxes)
            if len(rects_bboxes) > 1
            else rects_bboxes
        )
        for rect in merged_rects:
            pdf[page_num].add_redact_annot(rect, fill=fill_color)


def _group_lines_into_paragraphs(
    line_infos: List[LineInfo], line_gap_threshold: int
) -> List[Paragraph]:
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

        if (
            last_y is not None and (y0 - last_y) > line_gap_threshold
        ) or line_info.page_number != current_page_num:
            if current_paragraph:
                paragraphs_groups.append(current_paragraph)
            current_paragraph = []
            char_offset = 0
            paragraph_line_index = 0
            current_page_num = line_info.page_number

        temp_line_info = line_info._replace(
            line_start_char=char_offset,
            line_end_char=char_offset + len(line_info.text),
            paragraph_line_index=paragraph_line_index,
        )
        current_paragraph.append(temp_line_info)
        # Each line will be separated by a whitespace character.
        char_offset += len(temp_line_info.text) + 1
        last_y = y1
        paragraph_line_index += 1

    if current_paragraph:
        paragraphs_groups.append(current_paragraph)

    return [
        _create_paragraph(idx, group) for idx, group in enumerate(paragraphs_groups)
    ]


def _create_paragraph(index: int, lines: List[LineInfo]) -> Paragraph:
    text = " ".join([line_info.text for line_info in lines])

    return Paragraph(
        index=index, text=text, lines=lines, page_number=lines[0].page_number
    )



def _get_line_info(
    page_number: int,
    block_idx: int,
    line_idx: int,
    line_text: str,
    line_bbox: Tuple[float, float, float],
    spans: List[SpanInfo],
) -> LineInfo:
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
        spans_bbox=(
            (
                spans[0]["span_bbox"][0],
                spans[0]["span_bbox"][1],
                spans[-1]["span_bbox"][-2],
                spans_max_y,
            )
        ),
        text=line_text,
        spans=spans,
    )

    return line_info


def get_paragraphs(pdf: Document, line_gap_threshold: int = 5) -> List[Paragraph]:
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


def _get_pdf_span_boundaries(span: Dict) -> Rect:
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


def _get_span_info(
    span: dict, fill_color: Tuple[float, float, float], text_start: int
) -> SpanInfo:
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

    span_bbox = _get_pdf_span_boundaries(span)
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
        text_end=text_start + len(span_text),
    )

    return span_info


def _process_text_block(
    pdf_spans: Dict[str, List[SpanInfo]],
    page_number: int,
    block: Dict,
    block_idx: int,
    page_pixmap: Pixmap,
    zoom_level: int,
) -> None:
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
                fill_color = tuple(
                    c / 255.0
                    for c in page_pixmap.pixel(
                        int(line["bbox"][0] * zoom_level),
                        int(line["bbox"][1] * zoom_level),
                    )
                )

            for span in line["spans"]:
                span_info = _get_span_info(span, fill_color, text_start=len(line_text))
                line_text += span_info["text"]
                # For spans containing only whitespace characters the 'size' key won't be set.
                # if "size" in span_info:
                spans.append(span_info)
            if line_text != "":
                line_info = _get_line_info(
                    page_number, block_idx, line_idx, line_text, line["bbox"], spans
                )
                occurences = pdf_spans.get(line_text, [])
                occurences.extend([line_info])
                pdf_spans[line_text] = occurences


def _get_pdf_page_lines(page: Page) -> Dict[str, List[LineInfo]]:
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
        _process_text_block(
            pdf_spans, page.number, block, block_idx, page_pixmap, zoom_level
        )

    return pdf_spans


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
        lines = _get_pdf_page_lines(page)
        for line, line_infos in lines.items():
            if line in pdf_lines:
                pdf_lines[line].extend(line_infos)
            else:
                pdf_lines[line] = line_infos

    return pdf_lines


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
            images.append(
                ImageInfo(
                    xref=xref,
                    page_number=page.number,
                    ext=base_image["ext"],
                    data=Image.open(io.BytesIO(image_bytes)),
                )
            )

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


def get_image_rects(pdf: Document, page_number: int, img_xref: str) -> List[Rect]:
    """
    Returns boundary boxes and transformation matrices of an embedded image belonging 
    to the object with the given cross-reference number. 
    
    Args:
        pdf: PyMuPDF's Document object.
        page_number: PDF page number.
        img_href: the reference name entry of the item for which we need to 
                  return boundary boxes.
                
    Returns:
        a list of boundary boxes and respective transformation matrices for each image occurrence on the page,
        if the item is not on the page, an empty list is returned.
    """

    return pdf[page_number].get_image_rects(img_xref)


def draw_redact_rects(pdf: Document, redacts: List[TextRedactInfo]) -> None:
    """
    Draws redaction rectangle.

    Args:
        pdf: PyMuPDF Document object.
        redacts: a list of texts' redactions.

    Returns:
        None
    """

    for redact in redacts:
        pdf[redact["page_number"]].draw_rect(
            redact["draw_rect"], fill=redact["fill_color"]
        )


def apply_redactions(
    pdf: Document,
    text_redactions: List[TextRedactInfo],
    images_redactions: List[RedactInfo],
    drawings_redactions: List[RedactInfo],
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

    draw_redact_rects(pdf, text_redactions)


def save_processed_document(pdf: Document, output_path: str) -> None:
    """
    Stores the optimized version of the document, regarding its size, to the provided path.

    Args:
        pdf: PyMuPDF Document object.
        output_path: a path where the PDF document should be saved.
    """

    pdf.subset_fonts()
    pdf.ez_save(output_path.absolute().as_posix())
