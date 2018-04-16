# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:39:21 2018

@author: luiz_
"""


try:
    from adaline import Adaline
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from adaline import Adaline
    
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
            coeff.append([-1, float(row[1]), float(row[2]), float(row[3]),
                          float(row[4])])
            ans.append(int(row[5]))
        except ValueError as e:
            print(e)
    coeff = np.array(coeff)
    ans = np.array(ans)

    a = [Adaline(dimension=4, learning_rate=.0025) for i in range(5)]
    for neuron in a:
        neuron.train(coeff, ans)
        print("trained to get to:")
        print(neuron.weights)
    
    print("_________________________________________")
        
    f = open('evaluation.csv')
    csv_f = csv.reader(f, dialect='training')
    
    print("Evaluating samples:")
    for row in csv_f:
        row = [float(item) for item in row]
        print(row[0:], " \t=>",[neuron(row) for neuron in a])