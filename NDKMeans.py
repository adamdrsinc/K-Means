import random
import math
import GraphCalculations as gc
# Notes from Rich
# Do some calculations on how good the clustering is - silouhette, dunn index <- can use python library if needed, try not to
# Use matplotlib to generate graph for rich's viewing. Generate one 2D graph and one 4D graph


class NDKMeans:
    def __init__(self, coord_lists, k):
        self.k = k
        self.coord_lists = coord_lists
        self.coord_dimension_count = len(coord_lists[0])

        self.dimension_mins_maxes = []
        for i in range(self.coord_dimension_count):
            smallest = coord_lists[0][i]
            largest = coord_lists[0][i]
            for j in range(1, len(coord_lists)):
                current = coord_lists[j][i]
                if current < smallest:
                    smallest = current
                elif current > largest:
                    largest = current
            self.dimension_mins_maxes.append([smallest, largest])

        self.centroids = self.random_centroids()
        self.bins = self.make_bins()

    def inertia(self):
        inertia = 0
        for centroid, points in self.bins.items():
            for point in points:
                inertia += math.pow(gc.eucl_distance(point, centroid), 2)
        return inertia

    def make_random_location(self, centroids):
        unique = False
        while not unique:
            new_centroid = []
            for i in range(len(self.dimension_mins_maxes)):
                random_coord = random.uniform(
                    self.dimension_mins_maxes[i][0], self.dimension_mins_maxes[i][1])
                new_centroid.append(random_coord)
            if new_centroid not in centroids:
                unique = True
                return tuple(new_centroid)

    def random_centroids(self):
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

    def perform_kmeans(self):
        finished = False
        while not finished:
            # Grouping cluster with closest points
            for point in self.coord_lists:
                shortest_distance = None
                closest_centroid = None
                for centroid in self.centroids:
                    distance = gc.eucl_distance(point, centroid)
                    if shortest_distance == None:
                        shortest_distance = distance
                        closest_centroid = centroid
                    elif distance < shortest_distance:
                        shortest_distance = distance
                        closest_centroid = centroid

                self.bins[closest_centroid].append(point)

            # Getting averages of bins to set new clusters
            averages = []
            for _, data in self.bins.items():
                if len(data) == 0:
                    new_location = self.make_random_location(self.centroids)
                    averages.append(new_location)
                    continue

                data_average = []
                for i in range(self.coord_dimension_count):
                    dimension_total = 0
                    for coord in data:
                        dimension_total += coord[i]
                    data_average.append(dimension_total / len(data))
                averages.append(tuple(data_average))

            if (averages == self.centroids):
                finished = True
                return self.bins
            else:
                self.centroids = averages
                self.bins = self.make_bins()
