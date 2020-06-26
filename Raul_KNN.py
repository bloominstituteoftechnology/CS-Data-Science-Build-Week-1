import numpy as np
import scipy as sp
from scipy import stats

class Raul_KNN:
    """
    This class is used to predict the classification of a given data point
    based on a given dataset array. This is done by comparing the given
    point to it's K-Nearest-Neighbors in the dataset, where K is provided
    """
    def __init__(self, k):
        self.k = k
    
    # The fit method pulls the cleaned data into an array
    # and moves the classification feature to the last element in each row
    def knn_fit(self, data_array, j): # j is current row index of the classification feature
        """
        This method fits the data to the model
        """
        for row in data_array:
            row.append(row.pop(j))
        
        return data_array

    # We will be using euclidean distance as our distance measure, so we'll 
    # define a helper function 
    def euclid_dist(self, arow, brow):
        """
        This is a helper method for finding the euclidean distance
        """
        # We must instantiate our sum of squares of differences
        sumsqdiff = 0

        # We will iterate by index for each row. meaning that the differences 
        # between the unit vectors are found separately
        for i in range(0, len(arow)-1):
            sumsqdiff += (arow[i] - brow[i])**2

        # Now we return the squareroot
        return np.sqrt(sumsqdiff)

    
    def knn_predict(self, data_array, getrow):# data_array is the array produced by knn_fit
        """
        This function will predict the classification of a chosen point
        """
        # We will use euclidean distance to find the closest points(rows) in datarows to our getrow
        # We'll put the distances in a list 'eucdist'
        eucdist = []

        # And We will put the points in a list of arrays 'kneighbors'
        kneighbors = []

        # First let's find all the distances
        for row in data_array:
            eucdist.append(self.euclid_dist(getrow, row))
        
        # Now we have to find the smallest k-distances
        # Using numpy.partition is an efficient way of doing this
        kdists = []
        kdists = np.partition(eucdist, self.k-1)[:self.k]

        # Now we use kdists to compare with eucdist to find the kdists indices
        # We will use the indices of eucdist to find the appropriate rows from datarows
        for dist in kdists:
            # We'll use the index command to find the row from datarows that we want to append
            kneighbors.append(data_array[eucdist.index(dist)])        

        # Now we can predict the classification
        # Our prediction will be the most common classification in our found k neighbors
        classification = []
        for row in kneighbors:
            classification.append(row[-1])
        
        return (stats.mode(classification))[0]

# Simple test case
if __name__ == '__main__':

    point = [2.7810836,2.550537003]
    dataset = [
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]
        ]
    
    model = Raul_KNN(3)
    model.knn_fit(dataset, 2)

    print(point, ' is classified as:')
    print(model.knn_predict(dataset, point)) # Should be classified as [0]