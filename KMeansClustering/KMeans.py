import numpy as np
import random

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
    # fit function (self, data):
        # NOTE: some of the tasks below may be stored in helper
        #       functions in the actual implementation.
        # Initialize randomly selected centroids
            # Perhaps use random.choice()
        # Measure distances between each point and each cluster
            # Perhaps use np.linalg.norm()
                # This will ensure that regardless of number of
                # dimensions, the distance will still be calculable.
            # Store in appropriate data structure.
            # Assign point to nearest cluster.
                # Desired output:
                    # array, len(array) = len(centroids)
                    # contains all distances with index position
                    # of centroid.

        # Calculate the mean distance between each cluster
            # Fortunately, clusters is in the global scope of the
            # fit method, and its index position preserves the data
            # needed for retrieving the distance value.

            # These two factors make it possible to create
            # distance arrays by cluster.

            # Use cluster number in clusters and index pos in
            # clusters to refer back to appropriate values in
            # dist_dict to build arrays for each cluster.
                # First Pass may require a static solution.

            # Use these arrays to calculate the mean,
            # and reassign the centroids.

            # Repeat the above process with the mean distance
            # rather than the initial distance.

        # Calculate variation of each iteration:
            # Select clusters with the least amount of variation.

# For simplicity's sake, our First Pass will only calculate the
# initial iteration of the algorithm.


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        """
        input, 2D numpy array
        output, 
        """
        self.centroids = [tuple(random.choice(data)) for i in range(self.n_clusters)]
        dist = {}

        for centroid in self.centroids:
            distances = [np.linalg.norm(value - centroid) for value in data]
            dist[centroid] = distances

        print(dist, "\n")
            
        clusters = []
        for i in range(len(data)):
            comparison = []
            for j in range(len(self.centroids)):
                comparison.append(dist[tuple(self.centroids)[j]][i])

            cluster = comparison.index(min(comparison))
            clusters.append(cluster)

        print(clusters, "\n")
        print(list(dist.values()), "\n")

        for cluster in np.array(clusters):
            indicies = np.where(clusters == cluster)
            print(indicies)


random.seed(84)
data = np.array([
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
    [random.randint(0, 10), random.randint(0, 10)],
])
print(len(data))
print(data.shape)
print(f"{data}\n")

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(data)
print(kmeans)