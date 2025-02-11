"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # update weights if y != sin(w^T * x_i)
        # w = w + lr*x*y
        # b = b + lr*y
        # standardize features
        # early stopping to prevent accuracy from decreasing
        # learning rate decay

        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        self.w = np.random.uniform(-1, 1, (self.n_class, X_train.shape[1])) * 0.01
        self.b = np.random.uniform(-1, 1, (self.n_class,)) * 0.01

        initial_lr = self.lr
        all_acc = []
        best_accuracy = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(self.epochs):
            n_correct = 0

            self.lr = initial_lr / (1 + epoch * 0.1)

            for x, y in zip(X_train, y_train):
                scores = np.dot(self.w, x) + self.b
                y_pred = np.argmax(scores)
                if y != y_pred:
                    self.w[y] += self.lr * x
                    self.w[y_pred] -= self.lr * x
                    self.b[y] += self.lr
                    self.b[y_pred] -= self.lr
                else:
                    n_correct += 1
            
            curr_acc = n_correct / len(y_train)
            all_acc.append(curr_acc)
            if curr_acc > best_accuracy:
                best_accuracy = curr_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping at epoch", epoch, "with accuracy", curr_acc)
                    break
        print(all_acc)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
        scores = np.dot(X_test, self.w.T) + self.b
        return np.argmax(scores, axis=1)
