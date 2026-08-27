import math

def get_box_center(bbox):
    x1, y1, x2, y2 = bbox

    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    return x_center, y_center


def measure_distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)