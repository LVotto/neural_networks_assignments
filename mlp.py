# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:13:03 2018

@author: luiz_
"""

import numpy as np
import random
import matplotlib.pyplot as plt

import utils

class MLP:
    entries = []
    outputs = []
    samples = []
    desired = []
    training_error = 1
    deltas = {}
    
    def __init__(self, in_dim=1, out_dim=1, hidden=[],
                 specs=[], act_func=np.tanh,
                 learning_rate=.1, epsilon=1E-6, momentum=.8,
                 discrete=False, class_func=utils.sym_step):
        print(act_func)
        self.act_prime = np.vectorize(utils.diff(act_func))
        self.act_func = np.vectorize(act_func)
        
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discrete = discrete
        self.class_func = np.vectorize(class_func)
        
        if not specs:
            self.specs = [in_dim, *hidden, out_dim]
        else:
            self.specs = specs

        self.weight = []
        for layer in range(1, self.depth):
            self.weight.append([])
            for neuron in range(self.specs[layer]):
                self.weight[layer - 1].append([])
                for synapse in range(self.specs[layer - 1] + 1):
                    self.weight[layer - 1][neuron].append(
                            random.uniform(-1, 1)
                            )

    def __call__(self, entry):
        self._forward_propagate(entry)
        return self.outputs[-1][1:]
    
    @property
    def entry_size(self):
        return self.specs[0]

    @property
    def output_size(self):
        return self.specs[-1]
    
    @property
    def depth(self):
        return len(self.specs)

    def _forward_propagate(self, entry):
        self.entries = []
        self.outputs = []
        entry = [-1, *entry]
        self.entries.append([0 for n in range(self.specs[0] + 1)])
        self.outputs.append(entry)
        for layer in range(1, self.depth):
            self.entries.append([])
            self.outputs.append([-1])
            for neuron in range(self.specs[layer]):
                #print(self.weight[layer - 1][neuron])
                #print(self.outputs[layer - 1])
                self.entries[layer].append(
                        np.dot(self.weight[layer - 1][neuron],
                               self.outputs[layer - 1])
                        )
                self.outputs[layer].append(
                        self.act_func(self.entries[layer][neuron])
                        )
    
    def _deltas(self, layer):
        try:
            return self.deltas[layer]
        except:
            return []
    
    def _find_delta(self, layer, desired):
        delta = []
        if layer == self.depth - 1:
            out = np.array(self.outputs[layer][1:])
            delta = (np.add(desired[1:], -out) \
                     * self.act_prime(self.entries[layer]))
        else:
            next_delta = self.deltas[layer + 1]
            for neuron in range(self.specs[layer]):
                dw = 0
                for k in range(self.specs[layer + 1]):
                    dw += next_delta[k] * self.weight[layer][k][neuron]
                delta.append(dw * self.act_prime(self.entries[layer][neuron]))
        
        return delta
    
    def _update_weights(self, layer, desired):
        if layer == self.depth - 1:
            self.deltas = {}
        delta = self._find_delta(layer, desired)
        self.deltas[layer] = delta
        for j in range(self.specs[layer]):
            for i in range(self.specs[layer - 1]):
                increment = (self.learning_rate \
                         * delta[j] \
                         * self.outputs[layer - 1][i]) \
                         + self.momentum \
                         * (self.weight[layer - 1][j][i] \
                         - self._last_weight[layer - 1][j][i])
                self._last_weight[layer - 1][j][i] = self.weight[layer - 1][j][i]
                self.weight[layer - 1][j][i] += increment
            
    
    def store_training_set(self, samples, desired):
        self.samples = samples
        self.desired = desired
        if not (len(samples) or len(desired)):
            raise ValueError("""There must be values for samples to store.""")
    
    def _square_error(self, x, y):
        return 1 / 2 * (x - y) ** 2
    
    def train(self, samples=None, desired=None):
        if not self.samples:
            self.store_training_set(samples, desired)
        if len(desired[0]) == self.output_size:
            [d.insert(0, -1) for d in desired]
        self._last_weight = self.weight
        self.iterations = 0
        last_error = self.epsilon + 1
        error = 0
        errors = []
        while np.abs(error - last_error) > self.epsilon:
            last_error = error
            error = 0
            p = len(self.samples)
            for k in range(p):
                self._forward_propagate(self.samples[k])
                for u in range(self.output_size + 1):
                    out = self.outputs[-1][u]
                    error += self._square_error(self.desired[k][u], out)
                for i in range(1 - self.depth, 0):
                    self._update_weights(-i, self.desired[k])
            error *= 1 / p
            self.iterations += 1
            if not self.iterations % 50:
                print(self.iterations, '\t : ', error - last_error)
            errors.append(np.abs(error))
        print('done in', self.iterations, 'iterations.')
        print('With error = ', error, '.')
        self.training_error = error
        plt.plot(errors)
        plt.show()
        return self.iterations
            
if __name__ == "__main__":
    pass