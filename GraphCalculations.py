import math


def eucl_distance(data_point, cluster_point):
    # euclidean distance is defined as sqrt((x2​−x1​)^2 + (y2​−y1​)^2)

    total = 0
    for i in range(len(data_point)):
        total += math.pow(cluster_point[i] - data_point[i], 2)

    return math.sqrt(total)
