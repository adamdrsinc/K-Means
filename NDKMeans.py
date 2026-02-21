import random
import math
import GraphCalculations as gc


class NDKMeans:
    """
    NDKMeans is a library for performing K-means clustering on a given data set. NDKMeans takes the following
    hyperparameters:

    :param data: the data to perform K-means on. This data must be given as a 2D array, each array within
    representing a row of columns.
    :param k: the number of clusters desired.
    :param iteration_limit: the max number
    of times NDKMeans will try to reach convergence before continuing.
    """

    def __init__(self, data, k, iteration_limit):
        self.k = k
        self.iteration_limit = iteration_limit
        self.data = data
        self.coord_dimension_count = len(data[0])

        self.columnsMinAndMax = []

        # Getting the minimum and maximum of each column
        for column in range(self.coord_dimension_count):
            # Set smallest and largest to column's first row
            minimum = data[0][column]
            maximum = data[0][column]

            for row in range(0, len(data)):
                current = data[row][column]
                minimum = min(current, minimum)
                maximum = max(current, maximum)

            self.columnsMinAndMax.append({"minimum": minimum, "maximum": maximum})

        self.centroids = self.random_centroids()
        self.bins = self.make_bins()
        self.labels = []

    def inertia(self):
        inertia = 0
        for centroid, points in self.bins.items():
            for point in points:
                inertia += math.pow(gc.eucl_distance(point, centroid), 2)
        return inertia

    def make_random_location(self, centroids):
        """
        Makes a random coordinate for a centroid.

        :param centroids: A list of centroids.
        :returns: A tuple representing a new centroid.
        """
        unique = False
        while not unique:
            new_centroid = []
            for column in range(len(self.columnsMinAndMax)):
                random_coord = random.uniform(
                    self.columnsMinAndMax[column]["minimum"], self.columnsMinAndMax[column]["maximum"])
                new_centroid.append(random_coord)
            if new_centroid not in centroids:
                return tuple(new_centroid)

    def random_centroids(self):
        """
        Makes a new list of centroids.
        :returns: A new list of centroid tuples.
        """
        centroids = []
        for _ in range(self.k):
            new_location = self.make_random_location(centroids)
            centroids.append(new_location)

        return centroids

    def make_bins(self):
        bins = {}
        for centroid in self.centroids:
            bins[centroid] = []

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
                centroid_index = 0
                current_centroid_index = 0
                for centroid in self.centroids:
                    distance = gc.eucl_distance(point, centroid)
                    if shortest_distance is None or distance < shortest_distance:
                        shortest_distance = distance
                        closest_centroid = centroid
                        centroid_index = current_centroid_index
                    current_centroid_index += 1

                self.bins[closest_centroid].append(point)
                self.labels.append(centroid_index)

            # Getting averages of bins to set new clusters
            averages = []
            for _, data in self.bins.items():
                # If a centroid has no points closest to it, move it
                if len(data) == 0:
                    new_location = self.make_random_location(self.centroids)
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
                averages.append(tuple(data_average))

            # If the centroids don't move or the iteration count is reached, convergence is met
            if self.check_convergence(averages, current_iteration):
                convergence_met = True
            else:  # Otherwise set up for next iteration
                self.set_up_next_iteration(averages)

        return self.centroids
