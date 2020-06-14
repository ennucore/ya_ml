import numpy as np
from random import uniform


class Perceptron:
    def __init__(self):
        self.epochs = 12
        self.weights = np.array([uniform(0, 1) for _ in range(5)])

    def train(self, x, y):
        prediction = sum(x[i] * self.weights[i] for i in range(5))
        delta = y - prediction
        norm = np.linalg.norm(x)# [x[i] * self.weights[i] for i in range(5)])
        # print(delta, self.weights, norm, x)
        for i in range(5):
            self.weights[i] += (0.005 * delta * (x[i]) ** 0.5) # if self.weights[i] and norm else (delta * 0.005)
        # print(delta, self.weights)

    def predict(self, x):
        return round(sum(x[i] * self.weights[i] for i in range(5)))
