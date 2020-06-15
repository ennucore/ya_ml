import random
from linear import LinearModel


def child(model_1, model_2):
    new_model = LinearModel.random()
    for i, k1, k2 in zip(range(len(model_1.ks)), model_1.ks, model_2.ks):
        new_model.ks[i] = random.choice([k1, k2])
    new_model.b = random.choice([model_1.b, model_2.b])
    return new_model


class Genetic:
    def __init__(self, n=40):
        self.models = [LinearModel.random() for _ in range(n)]
        self.chosen_model = None

    def test_model(self, model, x_data, y_data):
        return sum(model.predict(x) == y for x, y in zip(x_data, y_data)) / len(y_data)

    def predict(self, x):
        return self.chosen_model.predict(x)

    def train_on_data(self, xs, ys):
        for _epoch in range(80):
            self.models = sorted(self.models, key=lambda model: -self.test_model(model, xs, ys))
            n = len(self.models)
            self.models = self.models[:n // 2]
            old_models = self.models.copy()
            for _ in range(n // 2):
                self.models.append(child(random.choice(old_models), random.choice(old_models)))
        self.chosen_model = self.models[0]
