import random


def child(model_1, model_2):
    new_model = GModel()
    for i, k1, k2 in zip(range(len(model_1.ks)), model_1.ks, model_2.ks):
        new_model.ks[i] = random.choice([k1, k2])
    new_model.b = random.choice(model_1.b, model_2.b)


class GModel:
    def __init__(self):
        self.ks, self.b = [random.uniform(-1, 1) for _ in range(5)], random.uniform(-1, 1)

    def predict(self, x):
        # todo
        return 1   # round(sum(int(self.ks) for i in range(len(self.ks))) / len(x))

    def mutate(self):
        for i in range(len(self.ks)):
            self.ks[i] += random.uniform(-0.05, 0.05)
            self.bs[i] += random.uniform(-0.05, 0.05)


class Genetic:
    def __init__(self, n=20):
        self.models = [GModel() for _ in range(n)]
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
