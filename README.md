# Instructions for Running the Program
To run the program, perform the following steps:
1.	Navigate to main.py
2.	Install the following python packages:
>  pandas, scikit-learn, matplotlib
3. Run main.py

NB: If you wish to use your own data set, NDKMeans requires that the data passed in be a 2D array, the inner arrays representing a row of the CSV file given. For example, if there is a data set with the following columns/rows:

| Weight | Speed | Distance |
|--------|-------|----------|
| 24.2   | 38.6  | 82.2     |
| 42.6   | 72.3  | 92.1     |
Then the 2D array passed in data must look like: [ [24.2, 38.6, 82.2], [42.6, 72.3, 92.1] ]


# K-Means Clustering Library
An algorithm for clustering data through K-Means. It is designed currently to work on 2D data.
This library contains the following:
- A K-Means clustering algorithm designed to work on n dimensional data;
- Inertia calculations;
- Elbow graphing through matplotlib;
- Silhouette scoring.
