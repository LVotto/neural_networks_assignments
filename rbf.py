# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:18:27 2018

@author: luiz_
"""

import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import random

import utils

class RBF:
    def __init__(self, specs=[2, 2, 1], learning_rate=.01, epsilon=1E-7,
                 act_func=(lambda x: x), act_prime=(lambda x: 1)):
        self.specs = specs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.act_func = act_func
        self.act_prime = act_prime
        
        self.weights = []
        # Entry layer has no weights
        self.weights.append([])
        self.weights.append([])
        for j in range(self.specs[1]):
            middle_weight = [random.uniform(-1, 1) for i in range(self.specs[0])]
            self.weights[1].append(middle_weight)
        self.weights.append([])
        for j in range(self.specs[2]):
            out_weight = [random.uniform(-1, 1) for i in range(self.specs[1] + 1)]
            self.weights[2].append(out_weight)
        
        self.variances = [1 for neuron in range(self.specs[1])]
        
    def __call__(self, sample):
        middle_output = [-1]
        for j in range(self.specs[1]):
            middle_output.append(self._make_pseudo_sample(sample, j))
        out_input = []
        for j in range(self.specs[2]):
            out_input.append(np.dot(self.weights[2][j], middle_output))
        return self.act_func(out_input)
            
    def _dist(self, a, b):
        a_ = np.array(a)
        b_ = np.array(b)
        return np.linalg.norm(b_ - a_)
    
    def _k_means(self):
        for neuron in range(self.specs[1]):
            self.weights[1][neuron] = self.samples[neuron][:]
                
        last_groups = []
        groups = [[] for neuron in range(self.specs[1])]
    
        while last_groups != groups:
            last_groups = groups
            groups = [[] for neuron in range(self.specs[1])]
            for k in range(len(self.samples)):
                dists = []
                for neuron in range(self.specs[1]):
                    dists.append(self._dist(self.weights[1][neuron],
                                            self.samples[k]))
                min_index = min(enumerate(dists), key=itemgetter(1))[0]
                groups[min_index].append(k)
            
            for neuron in range(self.specs[1]):
                for synapse in range(self.specs[0]):
                    self.weights[1][neuron][synapse] =  0
                    for k in groups[neuron]:
                        inc = self.samples[k][synapse]
                        self.weights[1][neuron][synapse] += inc
                self.weights[1][neuron] = np.array(self.weights[1][neuron])
                self.weights[1][neuron] *= 1 / len(groups[neuron])
        
        for neuron in range(self.specs[1]):
            self.variances[neuron] = 0
            for k in groups[neuron]:
                for syn in range(self.specs[0]):
                    inc = (self.samples[k][syn] \
                           - self.weights[1][neuron][syn]) ** 2
                    self.variances[neuron] += inc
            self.variances[neuron] *= 1 / len(groups[neuron])
                
    def _make_pseudo_sample(self, sample, neuron):
        g = 1
        for i in range(len(sample)):
            g *= utils.gauss(sample[i],
                             self.weights[1][neuron][i],
                             self.variances[neuron])
        return g
    
    def _make_pseudo_sample_set(self):
        pseudo_samples = []
        for k in range(len(self.samples)):
            pseudo_samples.append([-1])
            for neuron in range(self.specs[1]):
                pseudo_samples[k].append(
                        self._make_pseudo_sample(self.samples[k], neuron))
        return pseudo_samples
        
    
    def train(self, samples, desired, log=False):
            self.samples = samples
            self.desired = desired
            
            self._k_means()
            pseudo_samples = self._make_pseudo_sample_set()
            
            iteration = 0
            error = self.epsilon + 1
            last_error = 0
            errors = []
            while np.abs(error - last_error) > self.epsilon:
                last_error = error
                inputs = []  
                for k in range(len(self.samples)):
                    inputs.append([])
                    for neuron in range(self.specs[2]):
                        inputs[k].append(np.dot(self.weights[2][neuron],
                                         pseudo_samples[k]))
                    outputs = [self.act_func(x) for x in inputs]
                    for j in range(self.specs[2]):
                        delta = (self.desired[k][j] - outputs[k][j]) \
                                * self.act_prime(inputs[k][j])
                        for i in range(self.specs[1] + 1):
                            self.weights[2][j][i] += self.learning_rate \
                                                  * delta \
                                                  * pseudo_samples[k][i]
                error = 0
                for k in range(len(self.samples)):
                    for neuron in range(self.specs[2]):
                        inp = np.dot(self.weights[2][neuron],
                                     pseudo_samples[k])
                        output = self.act_func(inp)
                        error += (desired[k][neuron] - output) ** 2
                error /= len(self.samples)
                errors.append(error)
                if log:
                    if not iteration % 72:
                        print(iteration, ': ', error)
                iteration += 1
            if log:
                print('Network trained in %s iterations.' % iteration)
                print('With error: ', error)
                plt.plot(errors)
                plt.show()
            
            """
if __name__ == "__main__":
    ax = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    samples = []
    for x in ax:
        for y in ax:
            samples.append([x, y])
        
    desired = []
    for sample in samples:
        desired.append([np.sin(sample[0] * sample[1])])
    
    r = RBF(specs=[2, 2, 1])
    r.train(samples, desired)
"""
            
            