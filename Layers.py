from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    @abstractmethod
    def update(self, lr):
        pass


class Dense(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = np.random.default_rng().uniform(-.1, .1, (in_size, out_size))
        self.biases = np.random.default_rng().uniform(-.1, .1, (1, out_size))
        self.dw = None
        self.db = None
        self.x = None

    def forward(self, x):
        # (batch_size x out_size) = (batch_size x in_size) @ (in_size x out_size) + (1 x out_size)
        out = x @ self.weights + self.biases
        self.x = x

        return out

    def backward(self, grad):
        # (in_size x out_size) = (in_size x batch_size) @ (batch_size x out_size)
        self.dw = (self.x.T @ grad) / grad.shape[0]
        self.db = np.mean(grad, axis=0, keepdims=True)

        # (batch_size x in_size) = (batch_size x out_size) @ (out_size x in_size)
        out_grad = grad @ self.weights.T

        return out_grad

    def update(self, lr):
        self.weights = self.weights - lr * self.dw
        self.biases = self.biases - lr * self.db


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.e**(-x))

        self.out = out

        return out

    def backward(self, grad):
        return self.out * (1 - self.out) * grad

    def update(self, lr):
        pass


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = x
        cond = x <= 0
        out[cond] = 0

        self.out = out

        return out

    def backward(self, grad):
        g = self.out
        g[g > 0] = 1

        return g * grad

    def update(self, lr):
        pass


class BinaryCrossEntropy(object):
    def __init__(self):
        super().__init__()

    def loss(self, y, y_hat):
        bce = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        loss = np.mean(bce)

        return loss

    def backward(self, y, y_hat):
        grad = - (y / y_hat - (1 - y) / (1 - y_hat))
        return grad
