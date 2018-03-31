# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:58:19 2018

@author: luiz_
"""

try:
    from perceptron import Perceptron
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from perceptron import Perceptron
    
import numpy as np
import csv


if __name__ == "__main__":
    f = open('training.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f, dialect='training')
    coeff = []
    ans = []
    next(csv_f)
    for row in csv_f:
        try:
            coeff.append([-1, float(row[1]), float(row[2]), float(row[3])])
            ans.append(int(row[4]))
        except ValueError:
            pass
    coeff = np.array(coeff)
    ans = np.array([0 if item == -1 else 1 for item in ans])

    p = [Perceptron(dimension=3, learning_rate=.01) for i in range(5)]
    for neuron in p:
        neuron.train(coeff, ans)
        print("trained to get to:")
        print(neuron.weights)
    
    print("_________________________________________")
        
    f = open('evaluation.csv')
    csv_f = csv.reader(f, dialect='training')
    
    print("Evaluating samples:")
    for row in csv_f:
        row = [float(item) for item in row]
        print(row[0:], " \t=>",[neuron(row) for neuron in p])