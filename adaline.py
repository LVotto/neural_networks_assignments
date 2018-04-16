# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:32:14 2018

@author: luiz_
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from utils import sign, mean_square_error

class Adaline:
    def __init__(self, dimension=2, learning_rate=.0025, epsilon=1E-6):
        self.weights = [random.uniform(-1, 1) for i in range(dimension + 1)]
        self.dimension = dimension
        self.training_set = []
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def __call__(self, sample):
        if sample[0] != -1:
            sample.insert(0, -1)
        return sign(np.dot(self.weights, sample))
    
    def _delta_rule(self, answer_set, sample_set):
        gradient = 0
        for k in range(len(answer_set)):
            gradient += (answer_set[k] - np.dot(self.weights, sample_set[k])) \
                     * sample_set[k]
        self.weights = self.weights + self.learning_rate * gradient
        
    
    def train(self, sample_set, answer_set):
        self.training_set = (sample_set, answer_set)
        print("initialized with weights:")
        print(self.weights)
        iterations = 0
        diff = self.epsilon + 1
        error_graph = []
        while diff > self.epsilon and iterations < 1E6:
            iterations += 1
            last = mean_square_error(answer_set, self.weights, sample_set)
            error_graph.append(last)
            self._delta_rule(answer_set, sample_set)
            current = mean_square_error(answer_set, self.weights, sample_set)
            diff = np.abs(last - current)
        print('Done in %d iterations.' % iterations)
        plt.plot(error_graph)
        plt.xlabel("Iterações")
        plt.ylabel("Erro quadrático médio [u.a.]")
        plt.show()