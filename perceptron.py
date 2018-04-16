# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:49 2018

@author: luiz_
"""

import numpy as np

from utils import random_array, step, make_sample_set, plot_training_results


class Perceptron():
    training_set = None

    def __init__(self, dimension=2, learning_rate=.01):
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