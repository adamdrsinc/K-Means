from NDKMeans import NDKMeans
import GraphCalculations as gc
import pandas as pd
import matplotlib.pyplot as plot
import math


def show_elbow(data):
    inertia_scores = {}
    inertias = []
    ks = []
    for i in range(2, 10):
        ndkMeans = NDKMeans(data, i)
        ndkMeans.perform_kmeans()

        inertia = ndkMeans.inertia()

        inertia_scores[i] = inertia
        inertias.append(inertia)

        ks.append(i)

    for key, value in inertia_scores.items():
        print(f'K: {key}, Score: {value}')

    plot.plot(ks, inertias)
    plot.ylabel("Inertia")
    plot.xlabel("K Value")
    # plot.show()


def silouhette_score(ndkmeans):
    """
    1. For each data point, calculate two values:
    a. Average distance to all other data points within the same cluster
    b. Average distance to all data points within the nearest neighbouring cluster

    2. Compute the silhouette coefficient for each data point:
    = (1b - 1a) / max(1a, 1b)

    1. Calcualte the average silhouette coefficient
    """

    for centroid, points in ndkmeans.bins.items():
        # 1

        # 1. Calculate average distance to all other data points within the same cluster
        for i in range(len(points)):
            for j in range(i+1, len(points)):

                # 2

                # 1. Find the nearest neighbouring cluster to this data point
                # 2. Calculate the average distance to all data points in this neighbouring cluster from the
                #    current data point

                # 1. Calculate the mean intra-cluster distance
                # intracluster_total = 0
                # point_count = 0
                # for _, values in ndkmeans.bins.items():
                #     for i in range(len(values)):
                #         for j in range(i+1, len(values)):
                #             intracluster_total += gc.eucl_distance(values[i], values[j])
                #     point_count += len(values)
                # intracluster_average = intracluster_total / point_count

                # # 2. Calculate the mean nearest-cluster distance

                # nearestcluster_total = 0
                # for i in range(ndkmeans.k):
                #     for j in range(i+1, ndkmeans.k):
                #         nearestcluster_total += gc.eucl_distance(
                #             ndkmeans.centroids[i], ndkmeans.centroids[j])
                # nearestcluster_average = nearestcluster_total / ndkmeans.k

                # # Calculate silouhette (b - a) / max(a,b)
                # s_score = (nearestcluster_average - intracluster_average) / \
                #     max(intracluster_average, nearestcluster_average)

    return s_score


def main():
    customers = pd.read_csv("Mall_Customers.csv")
    data = customers[[
        'Age', 'Annual_Income_(k$)', 'Spending_Score']].values.tolist()

    show_elbow(data)

    ndkMeans = NDKMeans(data, 4)
    ndkMeans.perform_kmeans()
    print(silouhette_score(ndkMeans))
    # ndkMeans = NDKMeans(data, 5)
    # results = ndkMeans.perform_kmeans()


if __name__ == "__main__":
    main()
