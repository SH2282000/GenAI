import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm


def load_data(train=True):
    mnist = datasets.MNIST("../data", train=train, download=True)
    return mnist


def plot_examples(data: datasets.MNIST):
    """Plot some examples and put their corresponding label on top as title."""
    fig, axes = plt.subplots(3, 4, figsize=(8, 8))
    fig.suptitle("MNIST Dataset")

    for i in range(3):
        for j in range(4):
            index = np.random.randint(len(data))
            img, label = data[index]
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].set_title(f"{label=}")
            axes[i, j].axis("off")
    plt.show()


def report_stats(data: datasets.MNIST):
    print("Statistics of the train set:")
    print("Min:     ", torch.min(data.data))
    print("Max:     ", torch.max(data.data))
    print("Mean:    ", torch.mean(data.data.float()))
    print("Shape:   ", data.data.shape)
    print("Dtype:   ", data.data.dtype)


def convert_mnist_to_vectors(data):
    """Converts the ``[28, 28]`` MNIST images to vectors of size ``[28*28]``.
    It outputs mnist_vectors as a array with the shape of [N, 784], where
    N is the number of images in data.
    """

    mnist_vectors = []
    labels = []

    for image, label in tqdm(data):
        image_tensor = torch.Tensor(np.array(image))
        image_tensor = image_tensor.view(-1)

        mnist_vectors.append(image_tensor.numpy())
        labels.append(label)

    mnist_vectors = np.array(mnist_vectors)
    labels = np.array(labels)

    mnist_vectors = (mnist_vectors - 0.5) / 0.5

    # return as numpy arrays
    return mnist_vectors, labels


def do_pca(data):
    """Returns matrix [784x784] whose columns are the sorted eigenvectors.
    Eigenvectors (prinicipal components) are sorted according to their
    eigenvalues in decreasing order.
    """

    mnist_vectors, labels = convert_mnist_to_vectors(data)
    #     prepare_data(mnist_vectors)

    # compute covariance matrix of data with shape [784x784]
    cov = np.cov(mnist_vectors.T)

    # compute eigenvalues and vectors
    eigVals, eigVec = np.linalg.eig(cov)

    # sort eigenVectors by eigenValues
    sorted_index = eigVals.argsort()[::-1]
    eigVals = eigVals[sorted_index]
    sorted_eigenVectors = eigVec[:, sorted_index]
    print(type(sorted_eigenVectors), sorted_eigenVectors.shape)
    return sorted_eigenVectors.astype(np.float32).T


def plot_pcs(sorted_eigenVectors, num=10):
    """Plots the first ``num`` eigenVectors as images."""

    plt.figure(figsize=(10, 3))

    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(sorted_eigenVectors[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()


def plot_projection(sorted_eigenVectors, data):
    """Projects ``data`` onto the first two ``sorted_eigenVectors`` and makes
    a scatterplot of the resulting points"""
    mnist_vectors, labels = convert_mnist_to_vectors(data)
    projected_data = np.dot(mnist_vectors, sorted_eigenVectors[:, :2])

    plt.figure(figsize=(8, 6))
    for digit in range(10):
        indices = np.where(labels == digit)
        plt.scatter(
            projected_data[indices, 0], projected_data[indices, 1], label=str(digit)
        )
    plt.title("MNIST Dataset Projection in 2D Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Digit")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # You can run this part of the code from the terminal using python ex1.py
    # dataloading
    data = load_data()

    # subtask 1
    plot_examples(data)

    # subtask 2
    mnist_vectors, labels = convert_mnist_to_vectors(data)
    # Comment in once the above function is implemented, to check the shape of your dataset
    print("Data shape", mnist_vectors)

    # subtask 3
    pcs = do_pca(data)

    # subtask 3
    plot_pcs(pcs)

    # subtask 4
    plot_projection(pcs, data)
