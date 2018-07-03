# -*- coding: utf-8 -*-
"""
Created on Tue May 22 01:58:35 2018

@author: luiz_
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:45:14 2018

@author: luiz_
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:24:39 2018

@author: luiz_
"""

try:
    from tdnn import TDNN
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from tdnn import TDNN
import utils
    
import numpy as np
import csv
SPECS = {0: [5, 10, 1], 1: [10, 15, 1], 2: [15, 25, 1]}

def train_nets(n):
    global SPECS
    f = open('samples.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f,dialect='training')
    samples = []
    next(csv_f)
    for row in csv_f:
        try:
            samples.append([float(row[1])])
        except ValueError as e:
            print(e)
    
    samples = np.array(samples)
    nets = []
    for i in range(n):
        nets.append([TDNN(specs=SPECS[i],
                     act_func=utils.logistic) for j in range(3)])
    for topology in nets:
        for net in topology:
            net.train(series=samples)
            print("trained to get to:")
            print(net.weight)
    
    return nets, samples


if __name__ == "__main__":
    if do:
        ns = train_nets(3)
        nets = ns[0]
        samples = ns[1]
    do = False
    
    for topology in nets:
        for net in topology:
            net(120)
    
    for t in range(101, 121):
        print(t, end="\t: ", flush=True)
        for topology in nets:
            for net in topology:
                print(('%.3f' % float(net(t))), end="\t", flush=True)
        print('\n')

    f = open('evaluation.csv')
    csv_f = csv.reader(f,dialect='training')
    
    print("Evaluating samples:")
    rows = []
    for row in csv_f:
        if not row[0]:
            continue
        row = [float(item) for item in row[1:]]
        rows.append(row)
    d = [row for row in rows]
    for topology in nets:
        for net in topology:
            f = [float(net(t)) for t in range(101, 121)]
            plt.plot(range(101, 121), f, label='Rede')
            plt.plot(range(101, 121), d, label='Desejado')
            plt.legend()
            plt.show()
