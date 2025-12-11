import NDKMeans
import matplotlib.pyplot as plot

def plot_elbow(data, title):
    inertia_scores = {}
    inertias = []
    ks = []
    for i in range(2, 10):
        ndkMeans = NDKMeans(data, i, 300)
        ndkMeans.perform_kmeans()

        inertia = ndkMeans.inertia()

        inertia_scores[i] = inertia
        inertias.append(inertia)

        ks.append(i)

    plot.plot(ks, inertias)
    plot.title(title)
    plot.ylabel("Inertia")
    plot.xlabel("K Value")
    plot.show()

def plot_cardinality(ndkmeans):
    centroids = [i for i in range(0, len(ndkmeans.centroids))]
    points = []
    for _, values in ndkmeans.bins.items():
        points.append(len(values))
    
    plot.bar(centroids, points)
    plot.xlabel("Centroid")
    plot.ylabel("Number of Items")
    plot.title("Number of Items by Centroid")
    plot.show()