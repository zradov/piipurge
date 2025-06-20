from pymupdf import Rect
from PIL.ImageFile import ImageFile
from typing import NamedTuple, TypedDict, Tuple, List


class LinePos(NamedTuple):
    """
    Used as a key in dictionaries to describe position of a line in a PDF page.
    """

    page_number: int
    line_bbox: Tuple[int, int, int, int]


class EntitySubstitute(NamedTuple):
    """
    Used as a metadata object containing information about the specific entity type,
    its location in the text and its replacement value.
    """

    # Entity text value that need to be replaced.
    old_val: str
    # Replacement value for an entity string original value.
    new_val: str
    # The position of the starting character of the found original
    # entity string in a paragraph of text.
    start_char: int
    # The position of the ending character of the found original
    # entity string in a paragraph of text.
    end_char: int
    # The position of the starting character of the found original entity string,
    # relative to the starting position of the line containing it.
    sent_start_char: int
    # The position of the ending character of the found original entity string,
    # relative to the starting position of the line containing it.
    sent_end_char: int
    # The index of a line within all lines in a paragraph.
    line_index: int
    # The index of the paragraph containing the entity string.
    paragraph_index: int
    # a type of the entity
    ent_type: str


class SpanInfo(TypedDict):
    """
    Used as a metadata object for spans found in blocks of text.
    """

    # font size
    size: float
    # font name
    font: str
    # font color
    color: float
    # Background color
    fill_color: Tuple[float, float, float]
    # text opacity 0..255 (255 = fully opaque)
    alpha: 255
    # Origin of the first glyph.
    origin: Tuple[float, float]
    # Maximum of the font's glyph bboxes.
    bbox: Tuple[float, float, float, float]
    # Minimum of the font's glyph bboxes (doesn't contain surrounding
    # spaces between font's glyphs and the font's bbox).
    span_bbox: Rect
    text: str
    # Position of the first character in the span relative to the
    # span's position in the line that contains the span measured
    # in number of characters.
    text_start: int
    # text_start + SPANS_LENGTH
    text_end: int


class LineInfo(NamedTuple):
    """
    A metadata object that contains information a line of text in a paragraph.
    """

    # An index of a page in a PDF document.
    page_number: int
    # An index of a block of text lines, internally blocks of text lines are
    # retrieved using PyMuPDF's Textpage.extractBLOCKS() method.
    block_index: int
    # An index of the line in a paragraph.
    line_index: int
    # A line's bounding box coordinates.
    line_bbox: tuple[float, float, float, float]
    # A line's bounding box coordinates, reduced to match bounding boxes of spans that it contains.
    spans_bbox: tuple[float, float, float, float]
    # A text of the line.
    text: str
    # A list of spans that are inside the line.
    spans: List[SpanInfo]
    # Index of the line in a paragraph
    paragraph_line_index: int = -1
    # Index of the starting character of the line in a paragraph
    line_start_char: int = -1
    # Index of the ending character of the line in a paragraph
    line_end_char: int = -1


class LineSubstitute(NamedTuple):
    """
    Used as the container class for the EntitySubstitute and the SpanInfo classes.
    """

    # Position of the starting character in the line in a paragraph
    ent_start_char: int
    # Position of the ending character in the line in a paragraph
    ent_end_char: int
    # Old value that will be replaced.
    old_val: str
    # A list of spans whose text combined, makes the old value text.
    sub_spans: List[SpanInfo]  # Add the proper item type
    # Metadata about the entity.
    entity_substitute: EntitySubstitute
    # Metadata about the line.
    line_info: LineInfo


class SavedDrawingInfo(NamedTuple):
    # An absolute path to an image on a local file system.
    img_path: str
    # A cropped bounding box that contains only the drawing.
    crop_bbox: Tuple[float, float, float, float]
    # An index of the page, containing the drawing, in the PDF document.
    page_number: int


class SavedImageInfo(NamedTuple):
    # An absolute path to an image on a local file system.
    img_path: str
    # An unique cross-reference number of object in a PDF document.
    xref: Tuple[float, float, float, float]
    # An index of the page, containing the drawing, in the PDF document.
    page_number: int


class Paragraph(TypedDict):
    """
    Represents a block of text.
    """

    # The index of the paragraph in all paragraphs in a PDF document.
    index: int
    # The entire paragraph's text.
    text: str
    # A list of lines comprising a paragraph.
    lines: list
    # An index of the page, containing the drawing, in the PDF document.
    page_number: int


class ImageInfo(TypedDict):
    """
    Metadata about an image.
    """

    # An integer unique identification for an image in a PDF document.
    xref: str
    # An index of the page, containing the drawing, in the PDF document.
    page_number: int
    # image extension
    ext: str
    # image binary data
    data: ImageFile


class ImageTextInfo(NamedTuple):
    """
    Metadata about an image text.
    """

    # an image size
    image_size: Tuple[int, int]
    # lines of text in an image
    text_lines: List[str]


class RedactInfo(TypedDict):
    """
    Metadata about the redaction rectangle.
    """

    # An index of a page in a PDF document.
    page_number: int
    # redaction rectangle
    rect: Tuple[float, float, float, float]
    # redaction rectangle fill color.
    fill_color: Tuple[float, float, float]


class TextRedactInfo(RedactInfo):
    """
    Metadata about the text redaction rectangle.
    """

    # An actual redaction rectangle that need to be drawn.
    # Compared to the PyMuPDF redaction rectangle, its height is
    # set as low as possible in order to avoid issues with redaction
    # of text in overlapping lines.
    draw_rect: Tuple[float, float, float, float]
    # Information about the line containing the text that's being redacted.
    line_sub: LineSubstitute
