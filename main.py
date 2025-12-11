from NDKMeans import NDKMeans
import MLScoring as mls
import pandas as pd
import GraphPlotting as gp
from sklearn.metrics import davies_bouldin_score

def main():
    ## Red Wine Quality Data Set
    winequality_set_red = pd.read_csv("winequality-red.csv", sep=';')
    data = winequality_set_red[['fixed acidity','volatile acidity','citric acid']].values.tolist()

    gp.show_elbow(data, "Red Wine Quality Inertia by Cluster Count K")
    ndkmeans = NDKMeans(data, 4, 300)
    ndkmeans.perform_kmeans()
    print(davies_bouldin_score(data, ndkmeans.labels))
    print(f"Silhouette Score for red wine data set: {mls.silouhette_score(ndkmeans)}")

    ###

    ## White Wine Quality Data Set
    winequality_set_white = pd.read_csv("winequality-white.csv", sep=';')
    data = winequality_set_white[['fixed acidity','volatile acidity','citric acid']].values.tolist()

    gp.show_elbow(data, "White Wine Quality Inertia by Cluster Count K")
    ndkmeans = NDKMeans(data, 4, 300)
    ndkmeans.perform_kmeans()
    print(davies_bouldin_score(data, ndkmeans.labels))
    print(f"Silhouette Score for white wine data set: {mls.silouhette_score(ndkmeans)}")

if __name__ == "__main__":
    main()
