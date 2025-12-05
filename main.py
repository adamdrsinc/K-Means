import math
import random


def eucl_distance(data_point, cluster_point):
    # euclidean distance is defined as sqrt((x2​−x1​)^2 + (y2​−y1​)^2)

    total = 0
    for i in range(len(data_point)):
        total += math.pow(cluster_point[i] - data_point[i], 2)

    return math.sqrt(total)


def random_centroids(data_array, k):
    # k          = number of random centroids
    # data_array = the data being used

    centroids = set()
    for i in range(k):
        unique = False
        while not unique:
            random_x = random.randint(0, len(data_array[0]))
            random_y = random.randint(0, len(data_array))
            point = [random_x, random_y]
            if point not in centroids:
                unique = True
                centroids += point


def bins_setup(centroids):
    for i in range(len(centroids)):


def main():
    temp_data = [[1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0]],

    k = 3
    centroids = random_centroids(data_array=temp_data, k=k)
    bins = bins_setup(centroids)
    # Calculate euclidean distance between each data object (o) and each centroid (c)
    # 1. Go through each data point and calculate the eucl_distance between it and each cluster
    # 2. Put them in bins for each cluster that it's closest to

    # Cluster each data point with the cluster it is closest to


if __name__ == "__main__":
    main()
