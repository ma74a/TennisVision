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

# canditade_points_indices = [0, 2, 12, 13]
def get_closest_keypoint_index(player_foot_position, court_keypoints, canditade_points_indices):
    closet_distance = float("inf")
    canditade_point_index = canditade_points_indices[0]
    for keypoint_index in canditade_points_indices:
        # Get the actual (x, y) coordinates because cour_keypoints [x0, y0, x1, y1]
                    # court_keypoint(x)                  # court_keypoint(y)
        keypoint = (court_keypoints[keypoint_index*2], court_keypoints[keypoint_index*2+1])
        # calculate the distance according to Y coordinates.
        distance = abs(player_foot_position[1] - keypoint[1])

        if distance < closet_distance:
            closet_distance = distance
            canditade_point_index = keypoint_index

    return canditade_point_index


def get_height_of_box(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(pt1, pt2):
    return abs(pt1[0]-pt2[0]), abs(pt1[1]-pt2[1])