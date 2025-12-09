from NDKMeans import NDKMeans
import pandas as pd


def main():
    customers = pd.read_csv("Mall_Customers.csv")
    data = customers[[
        'Age', 'Annual_Income_(k$)', 'Spending_Score']].values.tolist()

    inertia_scores = {}
    for i in range(2, 10):
        ndkMeans = NDKMeans(data, i)
        ndkMeans.perform_kmeans()
        inertia_scores[i] = ndkMeans.inertia()

    for key, value in inertia_scores.items():
        print(f'K: {key}, Score: {value}')

    # ndkMeans = NDKMeans(data, 5)
    # results = ndkMeans.perform_kmeans()


if __name__ == "__main__":
    main()
