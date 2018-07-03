# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:59:10 2018

@author: luiz_
"""

import numpy as np

def sigmoid(x, beta=1000):
    return np.tanh(beta * x)

class Hopfield:
    def __init__(self, width, n_classes, act_func=sigmoid, epsilon=1E-6):
        self.width = width
        self.n_classes = n_classes
        self.act_func = act_func
        self.epsilon = epsilon
        
    def __call__(self, entry):
        v = entry[:].T
        old_v = np.zeros(v.shape)
        iteration = 0
        while self._dist(v, old_v) > self.epsilon:
            old_v = v[:]
            v = np.matmul(self.weight, old_v)
            v = self.act_func(v)
            iteration += 1
            if iteration > 1000000:
                print(self._dist(v, old_v))
                return False
        return v
    
    def _dist(self, a, b):
        return np.linalg.norm(a - b)
    
    def define_weights(self, classes):
        n = self.width
        p = self.n_classes
        weight = np.zeros([n, n])
        for z in classes:
            weight += z * z.T
        self.weight = weight / n - p / n * np.eye(n)
        