from NDKMeans import NDKMeans
import MLScoring as mls
import pandas as pd
import GraphPlotting as gp
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler


# INSTRUCTIONS FOR RUNNING THE PROGRAM CAN BE FOUND IN README.md

def evaluate_wine_csv(csv, graph_name, k):
    csv = pd.read_csv(csv, sep=';')
    winequality_set = pd.DataFrame(csv)

    # Dropping target variable "quality"
    winequality_set.drop(["quality"], inplace=True, axis=1)

    # Ensuring duplicates are not present
    winequality_set.drop_duplicates(inplace=True)

    # Scaling the data so all data is between 0-1.
    # Code adapted from (learnmodernpython, 2026)
    min_max_scaler = MinMaxScaler()
    winequality_set = min_max_scaler.fit_transform(winequality_set)
    # End reference
    winequality_set = pd.DataFrame(winequality_set)

    data = winequality_set.values.tolist()

    # Finding elbow of data to determine K
    gp.plot_elbow(data, graph_name)

    ndkmeans = NDKMeans(data, k, 300)
    ndkmeans.perform_kmeans()

    print(f"Davies-Bouldin Score: {davies_bouldin_score(data, ndkmeans.labels)}")
    print(f"Mean Silhouette Score: {mls.silhouette_score_average(ndkmeans)}")


# INSTRUCTIONS FOR RUNNING THE PROGRAM CAN BE FOUND IN README.md
def main():
    ## Red Wine Quality CSV from Wine Quality Data Set
    print("## Evaluating Red Wine Data Set Clustering ##")
    evaluate_wine_csv("winequality-red.csv", "Red Wine Quality Inertia by Cluster Count K", 5)

    print("")

    ## White Wine Quality CSV from Wine Quality Data Set. The same steps as above are performed.
    print("## Evaluating White Wine Data Set Clustering ##")
    evaluate_wine_csv("winequality-white.csv", "White Wine Quality Inertia by Cluster Count K", 4)


if __name__ == "__main__":
    main()
