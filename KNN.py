# Imports
import numpy as np

# Class
class k_nn:
    # Calc euclidean distance to compare neighbors
    # Fit - predict - display 
    def __init__(self, num_neighbors=5):
        """Init definition"""
        self.num_neighbors = num_neighbors

    # Euclidean distance
    def euclidean_distance(self, a, b):
        """Returns euclidean distance between rows"""
        euclidean_distance_sum = 0.0  # initial value
        for i in range(len(a)):
            euclidean_distance_sum += (a[i] - b[i]) ** 2
        """Subtract - square - add to euclidean_distance_sum"""
        euclidean_distance = np.sqrt(euclidean_distance_sum)
        return euclidean_distance

    # Fit k Nearest Neighbors
    def fit_knn(self, X_train, y_train):
        """Fits the model using training data. X_train and y_train inputs for func"""
        self.X_train = X_train
        self.y_train = y_train

    # Predict X for kNN
    def predict_knn(self, X):
        """Return predictions for X based on the fit X_train and y_train data"""
        # initialize prediction_knn as empty list
        prediction_knn = []
        for i in range(len(X)):
            # initialize euclidean_distance as empty list
            euclidean_distance = []
            for row in self.X_train:
                # find eucl_distance to X using
                # euclidean_distance() function call and append to euclidean_distance list
                euclidean_distance_sum = self.euclidean_distance(row, X[i])
                euclidean_distance.append(euclidean_distance_sum)
            neighbors = np.array(euclidean_distance).argsort()[: self.num_neighbors]
            # initialize dict to count class occurrences in y_train
            neighbor_count = {}
            for num in neighbors:
                if self.y_train[num] in neighbor_count:
                    neighbor_count[self.y_train[num]] += 1
                else:
                    neighbor_count[self.y_train[num]] = 1

            # max count labels to prediction_knn
            prediction_knn.append(max(neighbor_count, key=neighbor_count.get))
        return prediction_knn

    # display list of nearest_neighbors & euclidian dist
    def display_knn(self, x):
        """Inputs -- x // outputs a list w/ nearest neighbors and euclidean distance."""
        # initialize euclidean_distance as empty list
        euclidean_distance = []
        for row in self.X_train:
            euclidean_distance_sum = self.euclidean_distance(row, x)
            euclidean_distance.append(euclidean_distance_sum)
        neighbors = np.array(euclidean_distance).argsort()[: self.num_neighbors]
        # empty display_knn_values list
        display_knn_values = []
        for i in range(len(neighbors)):
            n_i = neighbors[i]
            e_dist = euclidean_distance[i]
            display_knn_values.append((n_i, e_dist))  # changed to list of tuples
        return display_knn_values
