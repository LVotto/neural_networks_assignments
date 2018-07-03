# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 00:56:56 2018

@author: luiz_
"""

import csv

with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([1, 2, 3])