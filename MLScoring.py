import GraphCalculations as gc

def silouhette_score(ndkmeans):
    """
    1. For each data point, calculate two values:
    a. Average distance to all other data points within the same cluster
    b. Average distance to all data points within the nearest neighbouring cluster

    2. Compute the silhouette coefficient for each data point:
    = (1b - 1a) / max(1a, 1b)

    1. Calcualte the average silhouette coefficient
    """

    def find_closest_centroid(point, this_points_centroid):
        shortest_distance = gc.eucl_distance(point, ndkmeans.centroids[0])
        closest_centroid = ndkmeans.centroids[0]
        for i in range(1, len(ndkmeans.centroids)):
            if(ndkmeans.centroids[i] == this_points_centroid):
                continue
            distance = gc.eucl_distance(point, ndkmeans.centroids[i])
            if distance < shortest_distance:
                shortest_distance = distance
                closest_centroid = ndkmeans.centroids[i]
        
        return closest_centroid
             
    silouhette_coefficient_total = 0
    point_total = 0
    for centroid, points in ndkmeans.bins.items():
        point_total += len(points)        
        # For each data point
        for i in range(len(points)):
            point = points[i]
            
            # 1. Calculate average distance to all other data points within the same cluster
            point_intracluster_distance_total = 0
            for j in range(len(points)):
                if i == j:
                    continue
                other_point = points[j]
                distance = gc.eucl_distance(point, other_point)
                point_intracluster_distance_total += distance
            point_intracluster_distance_avg = point_intracluster_distance_total / len(points)

            # 2. Find the nearest neighbouring cluster to this data point
            # 3. Calculate the average distance to all data points in this neighbouring cluster from the
            #    current data point
            closest_centroid = find_closest_centroid(point, centroid)

            point_extracluster_distance_total = 0
            closest_centroid_points = ndkmeans.bins[closest_centroid]
            for closest_centroid_point in closest_centroid_points:
                distance = gc.eucl_distance(point, closest_centroid_point)
                point_extracluster_distance_total += distance
            point_extracluster_distance_avg = point_extracluster_distance_total / len(closest_centroid_points)

            # print(f"\n intracluster_total: {point_intracluster_distance_total}\nintracluster_avg: {point_intracluster_distance_avg}\nextracluster_total: {point_extracluster_distance_total}\nextracluster_avg: {point_extracluster_distance_avg}\n")

            # 4. Calculate coefficient
            silouhette_coefficient = (point_extracluster_distance_avg - point_intracluster_distance_avg) / max(point_intracluster_distance_avg, point_extracluster_distance_avg)
            silouhette_coefficient_total += silouhette_coefficient

    silouhette_score = silouhette_coefficient_total / point_total
    return silouhette_score