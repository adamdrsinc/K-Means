from NDKMeans import NDKMeans
import MLScoring as mls
import pandas as pd
import GraphPlotting as gp

def main():
    winequality_set_red = pd.read_csv("winequality-red.csv", sep=';')
    data = winequality_set_red[['fixed acidity','volatile acidity','citric acid']].values.tolist()

    #show_elbow(data, "Red Wine Quality Inertia by Cluster Count K")
    #ndkmeans = NDKMeans(data, 4, 300)
    #ndkmeans.perform_kmeans()
    #print(f"Silhouette Score for red wine data set: {mls.silouhette_score(ndkmeans)}")

    winequality_set_white = pd.read_csv("winequality-white.csv", sep=';')
    data = winequality_set_white[['fixed acidity','volatile acidity','citric acid']].values.tolist()

    #gp.show_elbow(data, "White Wine Quality Inertia by Cluster Count K")
    ndkmeans = NDKMeans(data, 4, 300)
    ndkmeans.perform_kmeans()
    print(ndkmeans.labelled_array)
    #gp.plot_cardinality(ndkmeans)
    #print(f"Silhouette Score for white wine data set: {mls.silouhette_score(ndkmeans)}")

if __name__ == "__main__":
    main()
