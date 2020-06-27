import numpy as np
import scipy as sp
from scipy import stats
from sklearn import datasets
from numpy import random
from Raul_KNN import Raul_KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# We'll use the Iris dataset to test our class
iris_data = datasets.load_iris()

data_array = iris_data.data.tolist()

irisclass = iris_data.target.tolist()

# We want the target features in the training data
for i in range(0, len(data_array)):
    data_array[i].append(irisclass[i])

# Randomly choosing indices for test data
randidx = (np.random.choice(range(149), size=25, replace=False)).tolist()

train = data_array.copy()
test = []
testclass = []

# Pulling the random indices from the training data
for i in randidx:
    test.append(train[i])

# Adjusting the train, test and testclass lists
for element in test:
    train.remove(element)
    testclass.append(element[-1])
    element.pop()

# Now we have a test, testclass and train data set,
# we can make our predictions and see how we did
model = Raul_KNN(3)
model.knn_fit(train, 4)

preds = []
for row in test:
    preds.append(model.knn_predict(train, row))

# This gives us a list of arrays, so we'll make a flat list
predlist = []
for sub in preds:
    for item in sub:
        predlist.append(item)

print('Predicted classifications:\n', predlist)
print('Actual classifications:\n', testclass)
print('Raul_KNN gave and accuracy of:', accuracy_score(testclass, predlist))

# Now let's check to see how the sklearn KNNClassifier does
xtrain = []
ytrain = []
for item in train:
    xtrain.append(item[:-1])
    ytrain.append(item[-1])


k = 3
model2 = KNeighborsClassifier(n_neighbors=k)
model2.fit(xtrain, ytrain)

preds2 = (model2.predict(test)).tolist()

print('Predicted classifications:\n', preds2)
print('Actual classifications:\n', testclass)
print('sklearn KNN gave and accuracy of:', accuracy_score(testclass, preds2))

