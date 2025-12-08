import math
import random


def eucl_distance(data_point, cluster_point):
    # euclidean distance is defined as sqrt((x2​−x1​)^2 + (y2​−y1​)^2)

    total = 0
    for i in range(len(data_point)):
        total += math.pow(cluster_point[i] - data_point[i], 2)

    return math.sqrt(total)


def random_centroids(x_extremities, y_extremities, k):
    # k          = number of random centroids
    # data_array = the data being used

    centroids = []
    for _ in range(k):
        centroids.append(random_centroid(
            centroids, x_extremities, y_extremities,))

    return centroids


def random_centroid(centroids, x_extremities, y_extremities,):
    unique = False
    while not unique:
        random_x = random.uniform(x_extremities[0], x_extremities[1])
        random_y = random.uniform(y_extremities[0], y_extremities[1])
        point = (random_x, random_y)
        if point not in centroids:
            unique = True
            return point


def bins_setup(centroids):
    bins = {}
    for centroid in centroids:
        bins[centroid] = []

    return bins


def main():
    points = [(2, 4), (0, 3), (4, 1), (1, 0), (3, 3),
              (2, 1), (0, 2), (1, 4), (3, 0), (4, 2)]

    x_extremities = (0, 5)
    y_extremities = (0, 5)
    k = 3  # make this a hyper parameter
    centroids = random_centroids(x_extremities, y_extremities, k)
    bins = bins_setup(centroids)

    while True:
        # Calculate euclidean distance between each data object (o) and each centroid (c)
        # 1. Go through each data point and calculate the eucl_distance between it and each cluster
        # 2. Put them in bins for each cluster that it's closest to
        for point in points:
            shortest_distance = 100_000_000  # Picked to represent infinity
            chosen_centroid = None
            for centroid in centroids:
                distance = eucl_distance(point, centroid)
                if distance < shortest_distance and distance >= 0:
                    shortest_distance = distance
                    chosen_centroid = centroid

            bins[chosen_centroid].append(point)

        averages = []
        # Calculate the average X and Y values of each data point in
        # each cluster and set this centroid as this average
        for _, data in bins.items():
            # key = centroid
            # value = array of points that are closest to it
            if len(data) == 0:
                averages.append(random_centroid(
                    averages, x_extremities, y_extremities))
                continue

            x_total = y_total = 0
            for i in range(len(data)):
                x_total += data[i][0]
                y_total += data[i][1]
            x_average = x_total / len(data)
            y_average = y_total / len(data)
            averages.append((x_average, y_average))

        print(f"""
        Previous Centroids: {centroids}
        Averages: {averages}
        """)

        if (averages == centroids):
            print("Finished.")
            break
        else:
            centroids = averages
            bins = bins_setup(centroids)


if __name__ == "__main__":
    main()

# Notes from Rich
# Make it n-dimensional
# Make K a hyper parameter
# Do some calculations on how good the clustering is - silouhette, dunn index <- can use python library if needed, try not to
# everything is in one function - separate it out (make it your own library)
# Have dataset be a csv / like a csv
# Use matplotlib to generate graph for rich's viewing. Generate one 2D graph and one 4D graph
