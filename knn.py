import numpy as np


class KNNModel:
    def __init__(self, k=4):
        self.k = k
        self.data = [], []

    def train(self, x, y):
        self.data[0].append(np.array(x))
        self.data[1].append(y)

    def predict(self, x):
        x = np.array(x)
        neighbours = sorted(list(range(len(self.data[0]))), key=lambda i: np.linalg.norm(x[:3] - self.data[0][i][:3]))[:self.k]
        return round(sum(self.data[1][i] for i in neighbours) / self.k)

    def train_on_data(self, xs, ys):
        [self.train(x, y) for x, y in zip(xs, ys)]
