import os
import tempfile
from pymupdf import Point, Rect
from PIL import Image, ImageDraw
from ..schemas import SavedDrawingInfo
from typing import List, Optional, Tuple, Generator


def are_rectangles_overlapping(rect1: Rect, rect2: Rect, tolerance: float=0.5):
    """
    Checks if the two rectangles are overlapping. The two rectangles 
    overlap if at least half of the width and height of one rectangle, 
    minus the tolerance value, is over the other rectangle. 

    Args:
        rect1: the first rectangle.
        rect2: the second rectangle.
        tolerance: how much offset to tolerate when deciding whether 
                   the rectangles are overlapping or not.

    Returns:
        True if the rectangles overlap otherwise False.
    """
    
    rect1_half_width = (rect1[2] - rect1[0]) / 2
    rect2_half_width = (rect2[2] - rect2[0]) / 2
    rect1_half_height = (rect2[3] - rect2[1]) / 2
    rect2_half_height = (rect2[3] - rect2[1]) / 2
    rect1_cx = rect1[0] + rect1_half_width
    rect1_cy = rect1[1] + rect1_half_height
    rect2_cx = rect2[0] + rect2_half_width
    rect2_cy = rect2[1] + rect2_half_height

    if (
        abs(rect1_cx - rect2_cx) + tolerance
    ) < rect1_half_width + rect2_half_width and (
        abs(rect1_cy - rect2_cy) + tolerance
    ) < rect1_half_height + rect2_half_height:
        return True

    return False


def merge_intersecting_rects(rects: List[Rect]) -> List[Rect]:
    """
    Finds and merges overlapping rectangles.

    Args:
        rects: a list of rectangles.

    Returns:
        a list of rectangles where overlapping rectangles are merged into a single rectangle.
    """
    
    merged_rects = []

    if rects:
        r1 = rects[0]
        i = 1
        temp_rects = rects
        temp_rects2 = []

        while True:
            r2 = temp_rects[i]

            if are_rectangles_overlapping(r1, r2):
                r1 = [
                    min(r1[0], r2[0]),
                    min(r1[1], r2[1]),
                    max(r1[2], r2[2]),
                    max(r1[3], r2[3]),
                ]
            else:
                temp_rects2.append(r2)

            i += 1

            if i == len(temp_rects):
                merged_rects.append(r1)
                if len(temp_rects2) < 2:
                    if len(temp_rects2) == 1:
                        merged_rects.append(temp_rects2[0])
                    break
                r1 = temp_rects2[0]
                temp_rects = temp_rects2[1:]
                i = 0
                temp_rects2.clear()

    return merged_rects


def save_drawings(
    drawings: List,
    page_width: int,
    page_height: int,
    dest_dir: str = tempfile.gettempdir(),
) -> Generator[SavedDrawingInfo, None, None]:
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

    for page_num, page_drawings in drawings.items():
        image = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(image)

        for idx, drawing in enumerate(page_drawings):
            drawing_bbox = draw_paths(draw, drawing["items"])

            if drawing_bbox:
                cropped_image = image.crop(drawing_bbox)
                if 0 not in cropped_image.size:
                    img_name = f"{page_num}_{idx}.jpeg"
                    img_path = os.path.join(dest_dir, img_name)
                    cropped_image.save(img_path)

                    yield SavedDrawingInfo(img_path, drawing_bbox, page_num)


def draw_paths(
    draw: ImageDraw, paths: List
) -> Optional[Tuple[float, float, float, float]]:
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


def _update_bbox(
    min_x: float, min_y: float, max_x: float, max_y: float, *points: Point
) -> Tuple[float, float, float, float]:
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

    for x, y in points:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return min_x, min_y, max_x, max_y


def _update_bezier_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    points: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
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
        max(max_y, y0, y1, y2, y3),
    )
