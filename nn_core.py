from nn_loss import *
import numpy as np

class NeuralNetwork:
    def __init__(self, loss='mse'):
        self.layers = []
        self.loss = loss

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        if self.loss == 'mse':
            output_gradient = mse_loss_derivative(y, self.forward(X))
        elif self.loss == 'cross_entropy':
            output_gradient = cross_entropy_loss_derivative(y, self.forward(X))
        else:
            raise ValueError("Unsupported loss function")

        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, X, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            if self.loss == 'mse':
                loss = mse_loss(y, self.forward(X))
            elif self.loss == 'cross_entropy':
                loss = cross_entropy_loss(y, self.forward(X))
            print(f'Epoch {epoch}, Loss: {loss}')