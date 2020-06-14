import numpy as np
import csv

from random_model import RandomModel
from perceptron_model import Perceptron
from knn import KNNModel


def read_file(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)
        return lines[0], np.array(lines[1:])


def get_x_data(all_data, header):
    is_train = int('Survived' in header)
    x_data = [
        [
            int(row[1 + is_train]), int(row[3 + is_train] == 'male'), float(row[4 + is_train]) if row[4 + is_train] else 2,
            int(row[5 + is_train]), int(row[6 + is_train])
        ]
        for row in all_data
    ]
    return x_data


def normalize(array):
    for i in range(len(array[0])):
        maximum = max(row[i] for row in array)
        for j in range(len(array)):
            array[j][i] /= maximum


def get_y_data(all_data):
    return [int(row[1]) for row in all_data]


def get_train_data():
    header, all_data = read_file('data/train.csv')
    xs, ys = get_x_data(all_data, header), get_y_data(all_data)
    return xs[:3 * len(xs) // 4], ys[:3 * len(xs) // 4]


def get_test_data():
    header, all_data = read_file('data/train.csv')
    xs, ys = get_x_data(all_data, header), get_y_data(all_data)
    return xs[3 * len(xs) // 4:], ys[3 * len(xs) // 4:]


def test_model(model, x_data, y_data):
    return sum(model.predict(input_data) == right_answer
               for input_data, right_answer in zip(x_data, y_data)) / len(x_data)


def train_model(model, x_data, y_data):
    [[model.train(x, y) for x, y in zip(x_data, y_data)] for _epoch in range(getattr(model, 'epochs', 1))]


def train_and_test_model(model_cls):
    model = model_cls()
    train_model(model, *train_data)
    print(f'Result for {model_cls.__name__}: {test_model(model, *test_data)}')
    return model


train_data, test_data = get_train_data(), get_test_data()
normalize(train_data[0])
normalize(test_data[0])
train_and_test_model(RandomModel)
perceptron = train_and_test_model(Perceptron)
# print('Perceptron weights:', perceptron.weights)
train_and_test_model(KNNModel)
