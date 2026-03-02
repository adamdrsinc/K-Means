import random
import math


def eucl_distance(first_point, second_point):
    """
    Calculates the Euclidean distance between two points on a graph. Assumes data points are equal in dimensions.
    The data points provided must be lists.
    :param first_point: The first data point.
    :param second_point: The second data point.
    :return: The Euclidean distance.
    """
    # euclidean distance is defined as sqrt((x2​−x1​)^2 + (y2​−y1​)^2)

    total = 0
    for i in range(len(first_point)):
        total += math.pow(second_point[i] - first_point[i], 2)

    return math.sqrt(total)


class NDKMeans:
    """
    NDKMeans is a library for performing K-means clustering on a given data set. NDKMeans takes the following
    hyperparameters:

    :param data: the data to perform K-means on. This data must be given as a 2D array, each array within
    representing a row of data in a table.
    :param k: the number of clusters desired.
    :param iteration_limit: the max number
    of times NDKMeans will try to reach convergence before continuing.
    """

    def __init__(self, data, k, iteration_limit):
        self.k = k
        self.iteration_limit = iteration_limit
        self.data = data
        self.coord_dimension_count = len(data[0])
        self.centroids = self.initial_centroids()
        self.bins = self.make_bins()
        self.labels = []

    def inertia(self):
        inertia = 0
        for centroid_index, points in self.bins.items():
            centroid = self.centroids[centroid_index]
            for point in points:
                inertia += math.pow(eucl_distance(point, centroid), 2)
        return inertia

    def pick_random_location(self):
        generated = random.randint(0, len(self.data) - 1)
        picked_item = self.data[generated]
        return picked_item

    def initial_centroids(self):
        centroids = []
        for _ in range(self.k):
            centroid = self.pick_random_location()
            centroids.append(centroid)

        return centroids

    def make_bins(self):
        bins = {}
        for centroid_index in range(len(self.centroids)):
            bins[centroid_index] = []

        return bins

    def check_convergence(self, averages, current_iteration):
        return averages == self.centroids or current_iteration == self.iteration_limit

    def set_up_next_iteration(self, averages):
        self.centroids = averages
        self.bins = self.make_bins()
        self.labels = []

    def perform_kmeans(self):
        convergence_met = False
        current_iteration = 0
        while not convergence_met:
            current_iteration += 1
            # Grouping each point to the closest cluster
            for point in self.data:
                shortest_distance = None
                closest_centroid = None

                for i in range(len(self.centroids)):
                    distance = eucl_distance(point, self.centroids[i])
                    if shortest_distance is None or distance < shortest_distance:
                        shortest_distance = distance
                        closest_centroid = i

                self.bins[closest_centroid].append(point)
                self.labels.append(closest_centroid)

            # Getting averages of bins to set new clusters
            averages = []
            for _, data in self.bins.items():
                # If a centroid has no points closest to it, move it
                if len(data) == 0:
                    new_location = self.pick_random_location()
                    averages.append(new_location)
                    continue

                # Averaging out all points in a cluster to replace cluster's centroid coordinates with
                # averaged coordinates
                data_average = []
                for i in range(self.coord_dimension_count):
                    dimension_total = 0
                    for coord in data:
                        dimension_total += coord[i]
                    data_average.append(dimension_total / len(data))
                averages.append(data_average)

            # If the centroids don't move or the iteration count is reached, convergence is met
            if self.check_convergence(averages, current_iteration):
                convergence_met = True
            else:  # Otherwise set up for next iteration
                self.set_up_next_iteration(averages)

        return self.centroids
