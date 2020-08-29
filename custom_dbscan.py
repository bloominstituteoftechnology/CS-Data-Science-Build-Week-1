import numpy as np
import scipy as sp

def l2_distance(v1, v2):

    """takes 2 numpy arrays. returns the L2 distance between them."""
    return sum((v1-v2)**2)**0.5



class DB_SCAN:
    """noise points have label -1, unassigned points 0, cluster labels begin at 1."""
    # noise points have label -1, unassigned points 0, cluster labels begin at 1.

    # note that the original sk-learn DBSCAN estimator does not have a predict method
    # only fit and fit_predict. There might be a good definition of predicting on
    # new data for DBSCAN, but it primarily exists to cluster the existing data
    
    def __init__(self, max_radius, min_points, distance_func=l2_distance):

        self.max_radius = max_radius
        self.min_points = min_points
        self.distance_func = distance_func

    def get_neighborhood(self, point_set, point, radius):

        """find the indices of points within a given radius of a given point."""
        return np.array([index for index, element in enumerate(point_set) if ((self.distance_func(element, point_set[point]) <= radius) and (index != point))])

    def expand_cluster(self, point_ind, neighbors, label):

        """ Recursive method which expands the cluster until we have reached the border
        of the dense area (density determined by eps and min_samples) """
        
        self.labels[point_ind] == label

        # Iterate through neighbors - each "neighbor" is an index
        for neighbor in neighbors:
            
            if self.labels[neighbor] == -1: # if the point was labeled noise: we determined it didn't have enough neighbors to be a seed
                self.labels[neighbor] = label # but it is a neighbor, so it must be the boundary of a cluster

            elif self.labels[neighbor] == 0: # if the point has yet to be labeled:
                self.labels[neighbor] = label # label it with the cluster label
                
                # then get the neighbor's neighbors
                neighbors_of_neighbor = self.get_neighborhood(point_set=self.X, point=neighbor, radius=self.max_radius)

                if len(neighbors_of_neighbor) >= self.min_points: # if the neighbor has more neighbors than the threshold
                    self.expand_cluster(point_ind=neighbor, neighbors=neighbors_of_neighbor, label=label)


    def fit(self, X):
        self.X = X # input data
        self.labels = [0] * self.X.shape[0] # set all labels for each point to unassigned.
        
        cluster_id = 1 # the id of the current cluster we're adding points to. initialize at 1.

        for ii in range(0, self.X.shape[0]): # for each point in dataset:

            if self.labels[ii] == 0: # if the point's label is unassigned:

                # get the neighborhood for the point (not including the point itself.)
                neighbors = self.get_neighborhood(point_set=self.X, point=ii, radius=self.max_radius)

                if len(neighbors) < self.min_points: # if the number of neighbors is below the threshold, label it noise.
                    self.labels[ii] = -1
                else: # otherwise, start a cluster from the seed point.
                    self.expand_cluster(ii, neighbors=neighbors, label=cluster_id)
                    cluster_id += 1

    def fit_predict(self, X):

        self.fit(X)
        return self.labels
                


if __name__ == "__main__":

    pass