# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:49 2018

@author: luiz_
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def step(x):
    return 0 if x < 0 else 1


def random_array(size=1, interval=[-1, 1]):
    return np.array([random.uniform(*interval) for
                     i in range(size)])


def make_sample_set(n_samples=100,
                    dimension=2,
                    force_linear_separability=True,
                    max_weight=1000,
                    max_abs=100,
                    coefficients=None):
    if not coefficients:
        coefficients = random_array(size=dimension + 1,
                                    interval=[-max_weight, max_weight])

    samples = np.array([random_array(size=dimension + 1,
                                     interval=[-max_abs, max_abs]) for i in range(n_samples)])
    for array in samples:
        array[0] = -1

    if force_linear_separability:
        answers = np.vectorize(step)(np.dot(samples, coefficients))
    else:
        answers = np.array([random.randint(0, 1) for
                            i in range(dimension + 1)])

    return samples, answers, coefficients


def plot_training_results(perceptron, expected_coeffs=None):
    if not perceptron.is_trained:
        raise ValueError('Perceptron must be trained.')

    points = [t[1:] for t in perceptron.training_set[0]]
    answers = perceptron.training_set[1]
    min_x1 = min([x[0] for x in points])
    max_x1 = max([x[0] for x in points])
    [plt.scatter(*points[i],
                 c='b' if answers[i] else 'r',
                 marker='.') for i in range(len(points))]
    t = np.linspace(min_x1, max_x1, 1000)
    p_line = [(perceptron.weights[0] - perceptron.weights[1] \
                  * x) / perceptron.weights[2] for x in t]
    p_line = [x if min_x1 < x < max_x1 else None for x in p_line]
    plt.plot(t, p_line, label='perceptron', linewidth=4)
    if expected_coeffs is not None:
        e_line = [(expected_coeffs[0] - expected_coeffs[1] * \
                      x) / expected_coeffs[2] for x in t]
        e_line = [x if min_x1 < x < max_x1 else None for x in e_line]
        plt.plot(t, e_line, label='data_set', linestyle='--', linewidth=4)
    plt.legend()
    plt.show()

class Perceptron():
    training_set = None

    def __init__(self, dimension=2, learning_rate=1):
        self.dimension = dimension
        self.weights = random_array(size=dimension + 1, interval=[0,1])
        self.learning_rate = learning_rate
        
    def __call__(self, sample):
        if sample[0] != -1:
            sample.insert(0, -1)
        return step(np.dot(self.weights, sample))

    def _hebb_rule(self, answer, y, sample):
        self.weights = self.weights \
                       + self.learning_rate \
                       * (answer - y) * sample

    def train(self, sample_set, answer_set):
        self.training_set = (sample_set, answer_set)
        print("initialized with weights:")
        print(self.weights)
        iterations = 0
        error = True
        while error and iterations < 1E5:
            error = False
            iterations += 1
            for k in range(len(answer_set)):
                u = np.dot(self.weights, sample_set[k])
                y = step(u)
                if y != answer_set[k]:
                    self._hebb_rule(answer_set[k], y, sample_set[k])
                    error = True
        print('Done in %d iterations.' % iterations)

    @property
    def is_trained(self):
        return bool(self.training_set)

if __name__ == "__main__":
    p = Perceptron(dimension=2, learning_rate=.2)
    s = make_sample_set()
    p.train(s[0], s[1])
    print('Perceptron trained:')
    print('Based on coefficients: ', s[2])
    print('Got: ', p.weights)
    plot_training_results(p, expected_coeffs=s[2])