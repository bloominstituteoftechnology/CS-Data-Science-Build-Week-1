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
    
    # For the KNN model, there is no real fitting or training
    # KNN models require that the entire dataset be stored
    # The algorithm will then find the K number of nearest data points to 
    # point being tested, and a prediction is made based on these 'neighbors'


    # As such, the fit method would simply be to pull the cleaned data in an array
    # and to move the classification feature to the last element in each row

    def knn_fit(self, data_array, j): # j is current row index of the classification feature
        for row in data_array:
            row.append(row.pop(j))
        
        return data_array

    # We will be using euclidean distance as our distance measure, so we'll 
    # define a helper function 
    def euclid_dist(self, arow, brow):
        # The euclidean distance is the square root of the sum of the squares of the differences
        # ...say that 3 times fast
        # # Note: Through linear algebra, we see that this can be represented as the dot product of vectors
        # ...which leads us to a wonderfully simple representation sqrt(sum(x-y)^2)

        # So we instantiate our sum of squares of differences
        sumsqdiff = 0
        # We're going to iterate by index for each row
        # ie we're going to do each cardinal direction of each vector separately
        for i in range(0, len(arow)-1):
            sumsqdiff += (arow[i] - brow[i])**2

        # Now we return the squareroot
        return np.sqrt(sumsqdiff)

    
    def knn_predict(self, data_array, getrow):# data_array is the array produced by knn_fit

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

        # Now we can use our algorithm to predict a classification
        # Our prediction will be the most common classification in our found kneighbors
        # Our classification value has already been moved to the end of each row
        classification = []
        for row in kneighbors:
            classification.append(row[-1])
        
        return (stats.mode(classification))[0]

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