# k Nearest Neighbor Classifier Algorithm Implementation

## Required Packages:

    - numpy
    - scikit-learn (for testing)

## Usage:

### Imports

```py
from sklearn.datasets import load_iris
from k_nearest_neighbors import k_nearest_neighbors
```

### Test data:

```py
iris = load_iris()
data = iris.data
target = iris.target
```

### Model:

```py
# Instantiate model
classifier = k_nearest_neighbors(n_neighbors=10)

# Fit
classifier.fit_knn(data, target)

# Prediction
classifier.predict_knn([[1,2,3,4,5,6,7,8,9,10]])

# Nearest neighbors and euclidean distance (specified in n_neighbors)
classifier.display_knn([[1,2,3,4,5,6,7,8,9,10]])
```



### Part II

For the second part of this Build Week project, you'll be writing up a
HOW-TO blog entry that describes the algorithm, how to implement it, and
what it's useful for.

Your target audience should be other developers who haven't seen the
algorithm before.

There's no size limit, but a reader should be able to begin
implementation of the algorithm based on the information presented.

Post your entry to any blog site, either your own or a platform like
[Medium](https://medium.com/).