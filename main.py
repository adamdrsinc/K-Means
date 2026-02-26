from NDKMeans import NDKMeans
import MLScoring as mls
import pandas as pd
import GraphPlotting as gp
from sklearn.metrics import davies_bouldin_score


# INSTRUCTIONS FOR RUNNING THE PROGRAM CAN BE FOUND IN README.md

def evaluate_red_wine():
    winequality_set_red = pd.read_csv("winequality-red.csv", sep=';')

    # Obtaining columns of data set that I want to test
    data = winequality_set_red[
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values.tolist()

    # Finding elbow for the red wine quality data set
    gp.plot_elbow(data, "Red Wine Quality Inertia by Cluster Count K")

    # Performing K-means on what is perceived to be the elbow
    ndkmeans = NDKMeans(data, 4, 300)
    ndkmeans.perform_kmeans()

    # Evaluating the clustering
    print("## Evaluating Red Wine Data Set Clustering ##")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(data, ndkmeans.labels)}")
    print(f"Silhouette Score:{mls.silhouette_score_average(ndkmeans)}")


def evaluate_white_wine():
    winequality_set_white = pd.read_csv("winequality-white.csv", sep=';')
    data = winequality_set_white[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values.tolist()

    gp.plot_elbow(data, "White Wine Quality Inertia by Cluster Count K")
    ndkmeans = NDKMeans(data, 4, 300)
    ndkmeans.perform_kmeans()
    print("## Evaluating White Wine Data Set Clustering ##")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(data, ndkmeans.labels)}")
    print(f"Average Silhouette Score: {mls.silhouette_score_average(ndkmeans)}")


# INSTRUCTIONS FOR RUNNING THE PROGRAM CAN BE FOUND IN README.md
def main():
    ## Red Wine Quality CSV from Wine Quality Data Set
    evaluate_red_wine()

    print("")

    ## White Wine Quality CSV from Wine Quality Data Set. The same steps as above are performed.
    evaluate_white_wine()


if __name__ == "__main__":
    main()
