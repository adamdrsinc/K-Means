from NDKMeans import NDKMeans
import matplotlib.pyplot as plot


def plot_elbow(data, title):
    inertiaScores = []
    kValues = []

    # Trying different K values from 2-10.
    for i in range(2, 10):
        ndkMeans = NDKMeans(data, i, 300)
        ndkMeans.perform_kmeans()

        inertia = ndkMeans.inertia()
        inertiaScores.append(inertia)

        kValues.append(i)

    plot.plot(kValues, inertiaScores)
    plot.title(title)
    plot.ylabel("Inertia")
    plot.xlabel("K Value")
    plot.show()