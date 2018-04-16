# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:36:03 2018

@author: luiz_
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def step(x):
    return 0 if x < 0 else 1

def sign(x):
    return -1 if x < 0 else 1


def random_array(size=1, interval=[-1, 1]):
    return np.array([random.uniform(*interval) for
                     i in range(size)])


def make_sample_set(n_samples=30,
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
    
def mean_square_error(d, w, x):
    result = 0
    p = len(d)
    for k in range(p):
        result += pow(d[k] - np.dot(w, x[k]), 2)
    return result / p
    
    
