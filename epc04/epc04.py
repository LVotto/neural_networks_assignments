# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:24:39 2018

@author: luiz_
"""

try:
    from mlp import MLP
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from mlp import MLP
    
import numpy as np
import csv


if __name__ == "__main__":
    f = open('samples.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f,dialect='training')
    samples = []
    desired = []
    ev_samples = []
    ev_desired = []
    next(csv_f)
    for row in csv_f:
        try:
            samples.append([float(row[1]), float(row[2]), float(row[3])])
            desired.append([-1, float(row[4])])
        except ValueError as e:
            print(e)
    
    samples = np.array(samples)
    desired = np.array(desired)
    nets = [MLP(specs=[3, 10, 1]) for i in range(5)]
    for net in nets:
        net.train(samples=samples, desired=desired)
        print("trained to get to:")
        print(net.weight)
    
    print("_________________________________________")
        
    f = open('evaluation.csv')
    csv_f = csv.reader(f,dialect='training')
    
    print("Evaluating samples:")
    for row in csv_f:
        if not row[0]:
            continue
        row = [float(item) for item in row[1:]]
        print(row[0:], " \t=>",[net(row[:-1], row[-1]) for net in nets])