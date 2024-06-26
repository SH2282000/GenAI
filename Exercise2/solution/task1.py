import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

#create dataset
N = 1000
N_train = int(N*0.9) #use 90% for training
N_test = N - N_train #rest for testing
x, y = make_moons(n_samples=N, noise=0.2,random_state=0)
#split into train and test set
xtrain, ytrain = x[:N_train,...], y[:N_train,...]
xtest, ytest = x[N_train:,...], y[N_train:,...]

def show_dataset(X=xtrain, y=ytrain):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='Class 1')
    plt.title('The make_moons dataset.')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

class KNN:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None


    def fit(self, x, y):
        #fit routine
        self.x_train = x
        self.y_train = y

    def kneighbours(self, q):
        #return nearest neighbour indices and disßtances
        distances = np.sqrt(np.sum((q - self.x_train)**2, axis=1))
        indices = np.argsort(distances)[:self.k]

        return indices, distances[indices]

    def predict_single(self, q):
        #prediction function
        indices, _ = self.kneighbours(q)
        classes, counts = np.unique(self.y_train[indices], return_counts=True)

        return classes[np.argmax(counts)]

    def predict_batched(self, Q):
        # Prediction function for batch prediction
        predictions = []
        for q in Q:
            indices, _ = self.kneighbours(q)
            classes, counts = np.unique(self.y_train[indices], return_counts=True)
            predictions.append(classes[np.argmax(counts)])
        return np.array(predictions)


def compare_classifiers():
    # Our KNN
    knn = KNN(k=5)
    knn.fit(xtrain, ytrain)

    #  Sklearn KNN
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(xtrain, ytrain)

    knn_predictions = knn.predict_batched(xtest)
    sklearn_predictions = sklearn_knn.predict(xtest)

    # Check if both predictions are the same
    predictions_match = np.array_equal(knn_predictions, sklearn_predictions)
    print(f"{knn_predictions.tolist()=       }\n{sklearn_predictions.tolist()=   }")
    print("Predictions match:", predictions_match)

def show_decision_boundary(knn):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict the class labels for each point in the meshgrid
    Z = knn.predict_batched(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the dataset
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Decision boundary.")
    plt.show()

def test_model():
    for n_neighbors in [2**n for n in range(1, 10)]:
        knn = KNN(k=n_neighbors)
        knn.fit(xtrain, ytrain)
        knn_predictions = knn.predict_batched(xtest)

        # Results analyze:
        _, counts = np.unique(knn_predictions, return_counts=True)
        print(f"For {n_neighbors=} with class 0 = {counts[0]} elements and class 1 = {counts[1]} elements.")
        show_decision_boundary(knn)

def extreme_case():
    knn = KNN(k=len(xtrain))
    knn.fit(xtrain, ytrain)
    knn_predictions = knn.predict_batched(xtest)

    # Results analyze:
    classes, counts = np.unique(knn_predictions, return_counts=True)
    print(f"For {len(xtrain)=} with class {classes[0]} = {counts[0]} and no elements from class 1.")
    show_decision_boundary(knn)

def class_probs(y=ytrain):
    # Report class probabilities p(c) on the train set.
    classes, counts = np.unique(y, return_counts=True)

    return dict(zip(classes, counts/len(y)))

def calculate_volume(distances, k):
    radius = distances[:, k - 1]  # Distance to the k-th nearest neighbor
    return np.pi * radius**2

def calculate_probabilities(k, distances, labels, query_label):
    """Calculate p(xn) and p(xn|c) for a given query point and its k nearest neighbors."""
    num_neighbors = len(distances)
    total_samples = num_neighbors * np.pi * distances[:, k - 1]**2  # Volume V* for all neighbors
    class_samples = np.sum(distances[labels == query_label][:, k - 1]**2) * np.pi  # Volume V* for neighbors of the same class
    p_xn = k / total_samples
    p_xn_c = k / class_samples if class_samples > 0 else 0  # Prevent division by zero
    return p_xn, p_xn_c

def plot_probabilities(k_values):
    """Plot p(xn) and p(xn|c) for each value of k."""
    pass