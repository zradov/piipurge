def get_intersection_area(bbox1, bbox2):
    x0, y0, x1, y1 = bbox1
    xx0, yy0, xx1, yy1 = bbox2
    max_x0 = max(x0, xx0)
    max_y0 = max(y0, yy0)
    min_x1 = min(x1, xx1)
    min_y1 = min(y1, yy1)

    intersection_area = max(0, min_x1 - max_x0) * max(0, min_y1 - max_y0)

    return intersection_area


def get_iou(bbox1, bbox2):
    x0, y0, x1, y1 = bbox1
    xx0, yy0, xx1, yy1 = bbox2
    intersection_area = get_intersection_area(bbox1, bbox2)
    union_area = (x1 - x0) * (y1 - y0) + (xx1 - xx0) * (yy1 - yy0) - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area
