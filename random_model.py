import random


class RandomModel:
    def predict(self, *args):
        return random.randint(0, 1)

    def train(self, *args):
        pass

    def train_on_data(self, xs, ys):
        pass
