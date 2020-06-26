# Imports
# my model
from k_nearest_neighbors import k_nearest_neighbors

# sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
data = iris.data
target = iris.target

# Train/Test splits
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)


# Sklearn-learn KNN Classifier
# Instantiate model
clf = KNeighborsClassifier(n_neighbors=10)

# Fit
clf.fit(X_train, y_train)

# Prediction
predict = clf.predict(X_test)
print("******SKLEARN******")
print("Prediction", predict)

# Accuracy Score
print(f"Scikit-learn KNN classifier accuracy: {accuracy_score(y_test, predict)}")

# y_pred
y_pred = clf.predict([X_test[0]])
print("y_pred", y_pred)

##########################################################################

# k_nearest_neighbors (build model)
# Instantiate model
classifier = k_nearest_neighbors(n_neighbors=10)

# Fit
classifier.fit_knn(X_train, y_train)

# Prediction
predict = classifier.predict_knn(X_test)
print("******BUILD MODEL******")
print("Prediction", predict)

# Accuracy Score
print(f"Build k_nearest_neighbors model accuracy: {accuracy_score(y_test, predict)}")

# y_pred
y_pred = classifier.predict_knn([X_test[0]])
print("y_pred", y_pred)

# Neighbor index and euclidean distance
neighbors = classifier.display_knn(X_test[0])
print("Neighbors and their correscponding euclidian distances", neighbors)



