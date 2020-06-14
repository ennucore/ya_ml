class GModel:
    def __init__(self):
        self.k, self.b = 0, 0


class Genetic:
    def __init__(self, n=20):
        self.models = [GModel() for _ in range(n)]
