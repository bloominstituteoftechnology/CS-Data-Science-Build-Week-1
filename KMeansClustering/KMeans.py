import numpy as np

# Construct KMeans class with attributes n_clusters, n_iter:
  # n_clusters is the desired number of clusters
  # n_iter is the number of iterations the algorithm will go through
  # before it settles on its final cluster
  # NOTE: this is a simplification from the Scikit-Learn algorithm's
  #       implementation of the K-Means Cluster algorithm in which
  #       a tolerance and max number of iterations are set.
  #       Here we have simply left it up to the user to define the
  #       appropriate number of iterations and can be tuned as a
  #       hyper-parameter for each data set.

# Define Methods
  # def fit(self, data)
    # Store centroids and data in dictionary
      # centroid number is the key, the data is the value
    # Loop accross data to add data to the centoroid dictionary
  # predict

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        # self.n_iter = n_iter

    def fit(self, data):
        """
        input, data is a 2D NumPy array
        with each element in the first dimension analagous to the
        column of a Pandas DataFrame, and each element in the second
        dimension analagous to the individual row value of a given
        Pandas DataFrame column.

        output, dictionary
        with each key representing a cluster and each value containing
        the data assigned to that cluster
        """
        self.centroids = {}

        for i in range(self.n_clusters):
            self.centroids[i] = data[i]

        return self.centroids


kmeans = KMeans(n_clusters=2)
data = np.random.random((2, 5))
print(data)
kmeans = kmeans.fit(data)
print(f"\n{kmeans}")
