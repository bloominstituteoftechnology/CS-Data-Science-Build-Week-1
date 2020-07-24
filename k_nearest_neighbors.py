# Imports
import numpy as np


# Class
class k_nearest_neighbors:
    """Class of the K Nearest Neighbors Algorithm implementation"""

    """
    **Implementation**

    Method:
    - euclidean_distance(a, b):
        Returns euclidian distance of values

    - fit_knn(X_train, y_train):
        Fits model to training data

    - predict_knn(X):
        Returns predictions for X based on fitted model

    - display_knn(x)
        Returns list of nearest_neighbors + corresponding euclidian distance

    """

    # Initialization
    def __init__(self, n_neighbors=5):  # default neighbors to be returned
        """Init for algorithm"""
        self.n_neighbors = n_neighbors

    # Euclidian Distance
    def euclidean_distance(self, a, b):
        """
        Returns euclidian distance of values between row a and row b

        Inputs: a : int or float
                b : int or float

        Output: euclidian_distance : float

        """
        eucl_distance = 0.0  # initializing eucl_distance at 0

        for index in range(len(a)):
            """
            Based on: https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,metric%20as%20the%20Pythagorean%20metric.

            Calculation: Subtract b from a, square difference,
            add to eucl_distance.

            """
            eucl_distance += (a[index] - b[index]) ** 2

            euclidian_distance = np.sqrt(eucl_distance)

        return euclidian_distance

    # Fit k Nearest Neighbors
    def fit_knn(self, X_train, y_train):
        """
        Fits model to training data

        Inputs: X_train : array of int or float
                y_train : list or array of target

        Output: N/A pass along to predict_knn

        NOTE: I specifically choose not to include a "build-in"
        data split function, as it allows more flexibility in choosing
        data split methods according preference and/or data problem.

        """
        self.X_train = X_train
        self.y_train = y_train

    # Predict X for kNN
    def predict_knn(self, X):
        """
        Returns predictions for X based on fitted X_train and y_train data

        Inputs: X : list or array

        Output: prediction_knn : list of floats for each vector in X
        """

        # initialize prediction_knn as empty list
        prediction_knn = []

        # # initialize euclidian_distances as empty list
        # euclidian_distances = []

        for index in range(len(X)):  # Main loop iterating through len(X)

            # initialize euclidian_distances as empty list
            euclidian_distances = []

            for row in self.X_train:
                # for every row in X_train, find eucl_distance to X using
                # euclidean_distance() and append to euclidian_distances list
                eucl_distance = self.euclidean_distance(row, X[index])
                euclidian_distances.append(eucl_distance)

            # sort euclidian_distances in ascending order, and retain only k
            # neighbors as specified in n_neighbors (n_neighbors = k)
            neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

            # initialize dict to count class occurrences in y_train
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

            # max count labels to prediction_knn
            prediction_knn.append(max(count_neighbors, key=count_neighbors.get))

        return prediction_knn

    # Print/display list of nearest_neighbors + corresponding euclidian
    # distance
    def display_knn(self, x):
        """
        Inputs: x : vector x

        Output: display_knn_values : returns a list containing nearest
        neighbors and their correscponding euclidian distances
        to the vector x wrapped in tuples
        """

        # initialize euclidian_distances as empty list
        euclidian_distances = []

        # for every row in X_train, find eucl_distance to x
        # using euclidean_distance() and append to euclidian_distances list
        for row in self.X_train:
            eucl_distance = self.euclidean_distance(row, x)
            euclidian_distances.append(eucl_distance)

        # sort euclidian_distances in ascending order, and retain only k
        # neighbors as specified in n_neighbors (n_neighbors = k)
        neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

        # initiate empty display_knn_values list
        display_knn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_distances = euclidian_distances[index]
            display_knn_values.append(
                (neighbor_index, e_distances)
            )  # changed to list of tuples
        # print(display_knn_values)
        return display_knn_values
