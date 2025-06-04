
def are_rectangles_overlapping(rect1, rect2, tolerance=0.5):
    rect1_half_width = (rect1[2] - rect1[0]) / 2
    rect2_half_width = (rect2[2] - rect2[0]) / 2
    rect1_half_height = (rect2[3] - rect2[1]) / 2
    rect2_half_height = (rect2[3] - rect2[1]) / 2
    rect1_cx = rect1[0] + rect1_half_width
    rect1_cy = rect1[1] + rect1_half_height
    rect2_cx = rect2[0] + rect2_half_width
    rect2_cy = rect2[1] + rect2_half_height
    
    if (abs(rect1_cx - rect2_cx) + tolerance) < rect1_half_width + rect2_half_width and \
        (abs(rect1_cy - rect2_cy) + tolerance) < rect1_half_height + rect2_half_height:
        return True
    
    return False


def merge_intersecting_rects(rects):
    merged_rects = []
    
    if rects:
        r1 = rects[0]
        i = 1
        temp_rects = rects
        temp_rects2 = []

        while True:
            r2 = temp_rects[i]

            if are_rectangles_overlapping(r1, r2):
                r1 = [min(r1[0], r2[0]), 
                      min(r1[1], r2[1]),
                      max(r1[2], r2[2]), 
                      max(r1[3], r2[3])]
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