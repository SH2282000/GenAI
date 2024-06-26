import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def make_data(noise=0.2, outlier=1):
    prng = np.random.RandomState(0)
    n = 500

    x0 = np.array([0, 0])[None, :] + noise * prng.randn(n, 2)
    y0 = np.ones(n)
    x1 = np.array([1, 1])[None, :] + noise * prng.randn(n, 2)
    y1 = -1 * np.ones(n)

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1]).astype(np.int32)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.1, shuffle=True, random_state=0
    )
    xplot, yplot = xtest, ytest

    outlier = outlier * np.array([1, 1.75])[None, :]
    youtlier = np.array([-1])
    xtrain = np.concatenate([xtrain, outlier])
    ytrain = np.concatenate([ytrain, youtlier])

    return xtrain, xtest, ytrain, ytest, xplot, yplot

class LinearLeastSquares:
    # 1. Complete the fit and predict routine in task2.py. (2P)
    def fit(self, x, y):
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        self.theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

        return np.sign(x @ self.theta)

    def report_accuracy(self, x, y):
        y_predicted = self.predict(x)

        return np.mean(y_predicted == y)
    
    def show_decision_boundary(self, x, y, task):
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))

        # Predict the class labels for each point in the meshgrid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and the dataset
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"Decision boundary on the {task} set.")
        plt.show()
