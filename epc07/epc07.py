# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:22:10 2018

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
    from rbf import RBF
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from rbf import RBF
import utils
    
import numpy as np
import csv

if __name__ == "__main__":
    f = open('samples.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f,dialect='training')
    samples = []
    desired = []
    for row in csv_f:
        print(row)
        try:
            samples.append([float(row[1]), float(row[2])])
            desired.append([float(row[3])])
        except ValueError as e:
            print(e, row)
    s = samples[:]
    #samples = np.array(samples)
    #desired = np.array(desired)
    r = RBF(specs=[2, 2, 1])
    r.train(samples=samples, desired=desired)
    print(samples == s)

    f = open('evaluation.csv')
    csv_f = csv.reader(f,dialect='training')
    
    print("Evaluating samples:")
    rows = []
    for row in csv_f:
        if not row[0]:
            continue
        row = [float(item) for item in row]
        rows.append(row)
    eval_samples = [row[:-1] for row in rows]
    eval_desired = [row[2] for row in rows]
    
    post_proc = lambda x : -1 if x < 0 else 1
    
    print('sample \t\t\t desired \t y \t\t y_post')
    for k in range(len(eval_samples)):
        print(eval_samples[k], '\t:', eval_desired[k],
              '\t\t', "%.4f" % r(eval_samples[k])[0], '\t',
              post_proc(*r(eval_samples[k])))
              