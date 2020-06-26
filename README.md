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