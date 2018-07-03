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
    from mlp import MLP
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from mlp import MLP
import utils
    
import numpy as np
import csv

def train_nets(n):
    f = open('samples.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f,dialect='training')
    samples = []
    desired = []
    next(csv_f)
    for row in csv_f:
        try:
            samples.append([float(row[i]) for i in range(1, 5)])
            desired.append([-1, *[float(row[i]) for i in range(5, 8)]])
        except ValueError as e:
            print(e)
    
    samples = np.array(samples)
    desired = np.array(desired)
    nets = [MLP(specs=[4, 15, 3],
                discrete=True,
                act_func=utils.logistic) for i in range(n)]
    for net in nets:
        net.train(samples=samples, desired=desired)
        print("trained to get to:")
        print(net.weight)
    
    return nets


if __name__ == "__main__":
    if not nets:
        nets = train_nets(5)

    print("_________________________________________")

    f = open('evaluation.csv')
    csv_f = csv.reader(f,dialect='training')
    
    print("Evaluating samples:")
    rows = []
    for row in csv_f:
        if not row[0]:
            continue
        row = [float(item) for item in row[1:]]
        rows.append(row)
    for i in range(len(nets)):
        print("_________ NET ", i, "_________")
        j = 0
        success = 0
        for row in rows:
            full_ans = nets[i](row[:4], [-1, *row[5:-1]])
            ans = np.vectorize(utils.sym_step)(full_ans)
            succeeded = all([row[4:][x] == ans[x] for x in range(3)])
            print(j, '\t: ', [int(x) for x in row[4:]], list(ans),
                              ["%.2f" % float(x) for x in full_ans])
            j += 1
            if succeeded:
                success += 1 / len(rows)
        
        print(success)