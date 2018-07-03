# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 00:21:09 2018

@author: luiz_
"""

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

SPECS = [[3, 5, 1], [3, 10, 1], [3, 15, 1]]

if __name__ == "__main__":
    f = open('./samples.csv')
    csv.register_dialect('training', delimiter=';')
    csv_f = csv.reader(f,dialect='training')
    samples = []
    desired = []
    for row in csv_f:
        try:
            samples.append([float(row[x]) for x in range(1, 4)])
            desired.append([float(row[4])])
        except ValueError as e:
            print(e, row)
    s = samples[:]
    #samples = np.array(samples)
    #desired = np.array(desired)
    nets = []
    for spec in SPECS:
        nets.append([RBF(specs=spec) for i in range(3)])

    for i in range(1, 4):
        print('\n\n_____ TOPOLOGY %s _____\n' % i)
        for net in nets[i - 1]:
            net.train(samples=samples, desired=desired, log=True)
            
    f = open('evaluation.csv')
    csv_f = csv.reader(f,dialect='training')
    ev_samples = []
    ev_desired = []
    for row in csv_f:
        try:
            ev_samples.append([float(row[x]) for x in range(0,3)])
            ev_desired.append([float(row[3])])
        except ValueError as e:
            print(e, row)
    
    error = {}
    var = {}
    for topology in nets:
        for net in topology:
            error[net] = 0
            var[net] = 0
    
    with open('ans.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k in range(len(ev_samples)):
            sample = ev_samples[k]
            desired = ev_desired[k]
            line = []
            for topology in nets:
                for net in topology:
                    ans = net(sample)[0]
                    line.append("%.4f" % ans)
                    error[net] += desired - ans
                    var[net] += (desired - ans) ** 2
            spamwriter.writerow(line)
        line_e = []
        line_v = []
        for topology in nets:
            for net in topology:
                error[net] /= len(ev_samples)
                line_e.append('%.4f' % error[net])
                var[net] /= len(ev_samples)
                line_v.append('%.4f' % var[net])
        spamwriter.writerow(line_e)
        spamwriter.writerow(line_v)