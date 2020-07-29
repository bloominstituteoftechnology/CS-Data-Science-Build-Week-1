import random
import numpy as np
import scipy
from scipy.spatial.distance import cdist, euclidean
import sys


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

            # Use these arrays to calculate the mean distance in,
            # each centroid.

            # Calcuate Geometric Median of each cluster and reassign the centroids:
                # 

            # Repeat the above process with the mean distance
            # rather than the initial distance.

        # Calculate variation of each iteration:
            # Select clusters with the least amount of variation.
                # This will happen naturally with each itteration,
                # as the values converge to the center of the clusters.

        # return dictionary of values with cluster number as key,
            # and all data points as values.

# For simplicity's sake, our First Pass will only calculate the
# initial iteration of the algorithm.


class KMeans:
    def __init__(self, n_clusters, n_iter=10, random_state=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data):
        """
        input, 2D numpy array
        output, 1
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        self.centroids = [tuple(random.choice(data)) for i in range(self.n_clusters)]
        dist = {}
        for centroid in self.centroids:
            distances = [np.linalg.norm(value - centroid) for value in data]
            dist[centroid] = distances
        
        self.clusters = []
        for i in range(len(data)):
            comparison = []
            for j in range(len(self.centroids)):
                comparison.append(dist[tuple(self.centroids)[j]][i])

            cluster = comparison.index(min(comparison))
            self.clusters.append(cluster)

        self.clusters = np.array(self.clusters)

        self.avgs = []
        self.geo_meds = []
        self.cluster_dict = {}
        for cluster in set(np.array(self.clusters)):
            indicies = np.where(self.clusters == cluster)
            dist_list = [list(dist.values())[cluster][i] for i in indicies[0] if i in indicies[0]]
            self.avgs.append(sum(dist_list) / len(dist_list))

            cluster_list = np.array([data[i] for i in indicies[0] if i in indicies[0]])
            self.geo_meds.append(self.geometric_median(cluster_list))

            self.cluster_dict[cluster] = cluster_list

        # Though it would have been preferable to use recursion for
        # conciseness, the need to initialize random values at the
        # top of the method necessitated using a while loop.
        while self.n_iter >= 1:
            self.n_iter -= 1

            self.centroids = [tuple(geo_med) for geo_med in self.geo_meds]
            dist = {}
            for centroid in self.centroids:
                distances = [np.linalg.norm(value - centroid) for value in data]
                dist[centroid] = distances
                
            self.clusters = []
            for i in range(len(data)):
                comparison = []
                for j in range(len(self.centroids)):
                    comparison.append(dist[tuple(self.centroids)[j]][i])

                cluster = comparison.index(min(comparison))
                self.clusters.append(cluster)
                    
            self.avgs = []
            self.geo_meds = []
            self.cluster_dict = {}
            for cluster in set(np.array(self.clusters)):
                indicies = np.where(self.clusters == cluster)
                dist_list = [list(dist.values())[cluster][i] for i in indicies[0] if i in indicies[0]]
                self.avgs.append(sum(dist_list) / len(dist_list))
                cluster_list = np.array([data[i] for i in indicies[0] if i in indicies[0]])
                self.geo_meds.append(self.geometric_median(cluster_list))
                self.cluster_dict[cluster] = cluster_list

            self.clusters = np.array(self.clusters)

    def predict(self, data):
        distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]
        cluster = distances.index(min(distances))
        return cluster

    def geometric_median(self, X, eps=1e-5):
        """
        Code by Orson Peters
        Algorithm by Yehuda Vardi and Cun-Hui Zhang
        """
        y = np.mean(X, 0)
        
        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]
            
            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)
            num_zeros = len(X) - np.sum(nonzeros)
            
            if num_zeros == 0:
                y1 = T
            
            elif num_zeros == len(X):
                return y
                
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y
                
            if euclidean(y, y1) < eps:
                return y1
                
            y = y1

if __name__ == "__main__":
    print("Python version")
    print(sys.version)
    print("NumPy version:", np.__version__)
    print("SciPy version:", scipy.__version__)

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

    print("\ndata")
    print(data)
        
    kmeans = KMeans(n_clusters=2, n_iter=10)
    kmeans.fit(data)
    print("\nclusters")
    print(kmeans.clusters)
 
    pred_data = np.array([[random.randint(0, 10), random.randint(0, 10)]])
    print("\npred_data")
    print(pred_data)

    print("\nprediction")
    print(kmeans.predict(pred_data))