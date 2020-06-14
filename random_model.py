import random


class RandomModel:
    def predict(self, *args):
        return random.randint(0, 1)

    def train(self, *args):
        pass
