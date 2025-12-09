import math
import random

# Notes from Rich
# Make it n-dimensional
# Do some calculations on how good the clustering is - silouhette, dunn index <- can use python library if needed, try not to
# everything is in one function - separate it out (make it your own library)
# Use matplotlib to generate graph for rich's viewing. Generate one 2D graph and one 4D graph


class MyKMeans:
    def __init__(self, coord_list, k):
        self.k = k

        """
            NOTES
            1. coord_list needs to be a list of lists.
            2. You need a list of all the mins and maxes of each list in coord_list so that you can make random centroids
            3. set a dimension count
            4. From there go and fix the rest of your code to make it suit n-dimensions
        """

    def eucl_distance(data_point, cluster_point):
        # euclidean distance is defined as sqrt((x2​−x1​)^2 + (y2​−y1​)^2)

        total = 0
        for i in range(len(data_point)):
            total += math.pow(cluster_point[i] - data_point[i], 2)

        return math.sqrt(total)

    def make_random_location(self, centroids, coord_count):
        unique = False
        while not unique:
            rdm_coords_list = []

            rdm_x = random.uniform(self.widths[0], self.widths[1])
            rdm_y = random.uniform(self.heights[0], self.heights[1])
            new_centroid = (rdm_x, rdm_y)
            if new_centroid not in centroids:
                unique = True
                return new_centroid

    def random_centroids(self):
        centroids = []
        for _ in range(self.k):
            new_location = self.make_random_location(centroids)
            centroids.append(new_location)

        return centroids

    def make_bins(centroids):
        bins = {}
        for centroid in centroids:
            bins[centroid] = []

        return bins

    def perform_kmeans(self):
        if (self.x_column == None or self.y_column == None):
            return False

        centroids = self.random_centroids()
        bins = self.make_bins(centroids)

        # If the columns are not equally sized, trim the larger.
        if len(self.x_column) != len(self.y_column):
            largest_items = len(min(self.x_column, self.y_column, key=len))
            self.x_column = self.x_column[:largest_items]
            self.y_column = self.y_column[:largest_items]
        points = zip(self.x_column, self.y_column)

        finished = False
        while not finished:
            # Grouping cluster with closest points
            for point in points:
                shortest_distance = None
                closest_centroid = None
                for centroid in centroids:
                    distance = self.eucl_distance(point, centroid)
                    if shortest_distance == None:
                        shortest_distance = distance
                        closest_centroid = centroid
                    elif distance < shortest_distance:
                        shortest_distance = distance
                        closest_centroid = centroid

                bins[closest_centroid].append(point)

            # Getting averages of bins to set new clusters
            averages = []
            for _, data in bins.items():
                if len(data) == 0:
                    new_location = self.make_random_location(centroids)
                    averages.append(new_location)
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
                finished = True
            else:
                bins = self.make_bins(averages)
