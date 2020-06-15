import random


class LinearModel:
    def __init__(self, ks, b):
        self.ks, self.b = ks, b

    @classmethod
    def random(cls):
        self = cls([0] * 3, 0)
        self.ks, self.b = [random.uniform(-1, 1) for _ in range(5)], random.uniform(-1, 1)
        return self

    def predict(self, x):
        return int(sum(self.ks[i] * x[i] for i in range(5)) + self.b > 0)

    def mutate(self):
        for i in range(len(self.ks)):
            self.ks[i] += random.uniform(-0.05, 0.05)
            self.bs[i] += random.uniform(-0.05, 0.05)


