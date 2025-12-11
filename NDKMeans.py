import random
import math
import GraphCalculations as gc
# Notes from Rich
# Do some calculations on how good the clustering is - silouhette, dunn index <- can use python library if needed, try not to
# Use matplotlib to generate graph for rich's viewing. Generate one 2D graph and one 4D graph


class NDKMeans:
    def __init__(self, coord_lists, k, iteration_limit):
        self.k = k
        self.iteration_limit = iteration_limit
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
        self.labelled_array = []

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

    def check_convergence(self, averages, current_iteration):
        return averages == self.centroids or current_iteration == self.iteration_limit
    
    def set_up_next_iteration(self, averages):
        self.centroids = averages
        self.bins = self.make_bins()
        self.labelled_array = []

    def perform_kmeans(self):
        convergence_met = False
        current_iteration = 0
        while not convergence_met:
            current_iteration += 1
            # Grouping cluster with closest points
            for point in self.coord_lists:
                shortest_distance = None
                closest_centroid = None
                centroid_index = 0
                current_centroid_index = 0
                for centroid in self.centroids:
                    distance = gc.eucl_distance(point, centroid)
                    if shortest_distance == None or distance < shortest_distance:
                        shortest_distance = distance
                        closest_centroid = centroid
                        centroid_index = current_centroid_index
                    current_centroid_index += 1

                self.bins[closest_centroid].append(point)
                self.labelled_array.append(centroid_index)

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

            if self.check_convergence(averages, current_iteration):
                convergence_met = True
            else:
                self.set_up_next_iteration(averages)
        
        return self.centroids
        
