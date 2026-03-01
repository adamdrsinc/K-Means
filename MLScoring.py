from NDKMeans import eucl_distance


def silhouette_score_average(ndkmeans):
    """
    Calculates the mean silhouette score according to Rousseeuw, 1987.

    1. a(i) = average dissimilarity of i to all other objects of cluster A
    2. d(i, C) = average dissimilarity of i to all objects of C
    3. b(i) = minimum d(i, C), where C != A
    4. silhouette = b(i) - a(i) / max(a(i), b(i))

    All silhouettes are made and then averaged, which is returned.
    """

    def average_distance_inner(point, cluster, index_to_miss):
        total_distance = 0.0
        for i in range(len(cluster)):
            if i == index_to_miss:
                continue
            distance = eucl_distance(point, cluster[i])
            total_distance += distance

        avg_distance = total_distance / (len(cluster) - 1)
        return avg_distance

    # Find average inter-cluster distance - average distance between all cluster centroids
    def average_intercluster_distance(point, cluster):
        total_distance = 0.0
        for i in range(len(cluster)):
            distance = eucl_distance(point, cluster[i])
            total_distance += distance

        avg_distance = total_distance / (len(cluster) - 1)
        return avg_distance

    silouhette_coefficient_total = 0
    point_total = 0
    for centroid_index, points in ndkmeans.bins.items():
        centroid = ndkmeans.centroids[centroid_index]
        point_total += len(points)
        # For each data point
        for i in range(len(points)):
            point = points[i]

            # a(i) = average dissimilarity of i to all other objects of A
            avg_intracluster_distance = average_distance_inner(point, points, i)

            # b(i) = minimum d(i, C), where C != A
            shortest_distance = None
            for inner_index, inner_points in ndkmeans.bins.items():
                if centroid_index == inner_index:
                    continue
                avg_intercluster_distance = average_intercluster_distance(point, inner_points)
                if shortest_distance is None or avg_intercluster_distance < shortest_distance:
                    shortest_distance = avg_intercluster_distance

            # silhouette = b(i) - a(i) / max(a(i), b(i))
            coefficient = (shortest_distance - avg_intracluster_distance) / max(avg_intracluster_distance,
                                                                                shortest_distance)
            silouhette_coefficient_total += coefficient

    return silouhette_coefficient_total / point_total
