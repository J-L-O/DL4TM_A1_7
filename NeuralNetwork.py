import math

import numpy as np
import tensorflow as tf
from Layers import Sigmoid, ReLU, Dense, BinaryCrossEntropy, Layer

ActivationFunctions = {
    'sigmoid': Sigmoid,
    'relu': ReLU
}


class NeuralNetwork(object):
    def __init__(self, layers: list, activations: list):
        self.layers = []
        self.loss = BinaryCrossEntropy()

        for i in range(len(activations)):
            in_size = layers[i]
            out_size = layers[i + 1]

            layer = Dense(in_size, out_size)
            self.layers.append(layer)

            activation = self.__class__.getActivationFunction(activations[i])
            self.layers.append(activation)

    def _feedforward(self, x):
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def _backpropagation(self, y, y_hat):
        gradient = self.loss.backward(y, y_hat)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def _update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def train(self, x, y, batch_size=10, epochs=100, lr=0.01):
        batches_per_epoch = math.ceil(x.shape[0] / batch_size)

        for epoch in range(epochs):
            print(f"Starting epoch {epoch}/{epochs}")
            for batch in range(batches_per_epoch):
                x_batch = x[batch_size * 10:(batch_size + 1) * 10]
                y_batch = y[batch_size * 10:(batch_size + 1) * 10]

                y_hat = self._feedforward(x_batch)

                loss = self.loss.loss(y_batch, y_hat)
                print(f"Batch {batch}/{batches_per_epoch} Loss {loss}")

                self._backpropagation(y_batch, y_hat)

                self._update(lr)

    def predict(self, x, y):
        y_hat = self._feedforward(x)
        y_hat = np.round(y_hat)
        acc = np.sum(y == y_hat) / y.shape[0]

        print(f"Accuracy: {acc}")

    @staticmethod
    def getActivationFunction(name: str) -> Layer:
        activation = ActivationFunctions[name]
        return activation()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    select_train = (y_train == 0) | (y_train == 1)
    select_test = (y_test == 0) | (y_test == 1)
    X_train = X_train[select_train] / 255
    y_train = np.expand_dims(y_train[select_train], 1)
    X_test = X_test[select_test] / 255
    y_test = np.expand_dims(y_test[select_test], 1)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    nn = NeuralNetwork([X_train.shape[1], 1000, 100, 1], activations=['relu', 'relu', 'sigmoid'])
    nn.train(X_train, y_train, epochs=10, batch_size=64, lr=0.0001)
    nn.predict(X_train, y_train)
    nn.predict(X_test, y_test)
