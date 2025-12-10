from NDKMeans import NDKMeans
import MLScoring as mls
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

    # for key, value in inertia_scores.items():
    #     print(f'K: {key}, Score: {value}')

    plot.plot(ks, inertias)
    plot.ylabel("Inertia")
    plot.xlabel("K Value")
    plot.show()


def main():
    customers = pd.read_csv("Mall_Customers.csv")
    data = customers[[
        'Age', 'Annual_Income_(k$)', 'Spending_Score']].values.tolist()

    show_elbow(data)

    ndkMeans = NDKMeans(data, 4)
    ndkMeans.perform_kmeans()
    print(f"Silouhette Score: {mls.silouhette_score(ndkMeans)}")


if __name__ == "__main__":
    main()
