import numpy as np


class KNNModel:
    def __init__(self, k=4):
        self.epochs = 1
        self.k = k
        self.data = [], []

    def train(self, x, y):
        self.data[0].append(np.array(x))
        self.data[1].append(y)

    def predict(self, x):
        x = np.array(x)
        neighbours = sorted(list(range(len(self.data[0]))), key=lambda i: np.linalg.norm(x - self.data[0][i]))[:self.k]
        return round(sum(self.data[1][i] for i in neighbours) / self.k)
