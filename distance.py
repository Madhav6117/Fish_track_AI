import math


def calculate_distance(previous_point, current_point):
    if previous_point is None or current_point is None:
        return 0

    dx = current_point[0] - previous_point[0]
    dy = current_point[1] - previous_point[1]

    return math.sqrt(dx * dx + dy * dy)