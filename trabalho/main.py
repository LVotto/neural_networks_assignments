# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:11:33 2018

@author: luiz_
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
import time

try:
    from mlp import MLP
    import utils
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from mlp import MLP
    import utils

if __name__ == '__main__':
    f = open('training_set.pickle', 'rb')
    sets = pickle.load(f)
    samples = sets['sample']
    desired = sets['desired']
    print(len(samples))
    
    print('Training your net. Please be patient and don\'t kill yourself yet.')
    print('This is the 120-20-2 net.')
    nn = MLP(specs=[120, 20, 8, 2], act_func=utils.logistic)
    t0 = time.time()
    nn.train(samples=samples, desired=desired)
    tf = time.time()
    print('Training done in %s seconds.' % (tf - t0))