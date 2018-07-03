# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:58:43 2018

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
    file = open('./test_set.pickle', 'rb')
    test_set = pickle.load(file)
    file.close()
    samples = test_set['sample']
    desired = test_set['desired']
    
    # Net 1 -> 120 -> 20 -> 2 ->
    file = open('./nn_120_20__2.pickle', 'rb')
    net1 = pickle.load(file)
    file.close()
    # Net 2 -> 120 -> 10 -> 8 -> 2 ->
    file = open('./net_120_10_2.pickle', 'rb')
    net2 = pickle.load(file)
    file.close()
    nets = [net1, net2]
    
    i = 0
    N = len(samples)
    success = {net: 0 for net in nets}
    r = {net: [] for net in nets}
    for i in range(N):
        for net in nets:
            result = net(samples[i])
            result = [int(round(np.linalg.norm(result[0]))),
                      int(round(np.linalg.norm(result[1])))]
            if result == desired[i]:
                success[net] += 1
            r[net].append(result)
            
    i = 0
    for s in success:
        success[s] = success[s] / N
        print('Net %s: %s success rate' % (i, success[s]))
        i += 1
    
    i = 0
    fem = {net: 0 for net in nets}
    for net in nets:
        for result in r[net]:
            if result == [1, 0]:
                fem[net] += 1
                