import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt


#load dataset
data = load_digits()
x, y = (data.images / 16.0).reshape(-1, 8 * 8), data.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=0)


#initialize parameters
weights = np.ones((1,10,x.shape[1])) #np.random.normal(0,1,size=(1,10,x.shape[1]))
bias = np.zeros((1,10))

lr = 0.01 # This is the learning rate. Source: trust me bro.
num_iterations = 5 # you choose :)

def calc_logits(x):
    return np.matmul(x, weights.transpose(0, 2, 1)) + bias

def calc_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
    
    return exp_logits / np.sum(exp_logits, axis=2, keepdims=True) # (n_samples, 1, 10)

def calc_gradients(logits):
    print("Shape of xtrain:", xtrain.shape)
    softmax_output = calc_softmax(logits)[0]
    print("Shape of softmax_output:", softmax_output.shape)
    one_hot_labels = np.eye(10)[ytrain].reshape(-1, 1, 10).squeeze()  # Labels' one-hot encoding
    print("Shape of one_hot_labels:", one_hot_labels.shape)

    gradient = -np.mean(xtrain * (one_hot_labels - softmax_output), axis=0)  # (1, 10, 64)
    gradient_bias = -np.mean(one_hot_labels - softmax_output, axis=0) # (1, 10)
    

    return gradient, gradient_bias

def calc_accuracy(x):
    logits = calc_logits(x)
    predictions = np.argmax(logits, axis=2).flatten()

    return np.mean(predictions == ytrain)

def train():
    train_accuracies = []
    test_accuracies = []

    for i in range(num_iterations):
        logits = calc_logits(xtrain)
        # gradients
        gradient, gradient_bias = calc_gradients(logits)
        
        #optimization
        weights -= lr * gradient
        bias -= lr * gradient_bias
        
        # Accuracy testing
        # TRAIN set
        train_accuracies.append(calc_accuracy(xtrain))
        # TEST set
        test_accuracies.append(calc_accuracy(xtest))
        
        print(train_accuracies)
        print(test_accuracies)