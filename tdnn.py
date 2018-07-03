# -*- coding: utf-8 -*-
"""
Created on Tue May 22 01:38:20 2018

@author: luiz_
"""

import numpy as np

from mlp import MLP


class TDNN(MLP):
    timeline = []
    
    def __init__(self, *args, **kwargs):
        MLP.__init__(self, *args, **kwargs)
    
    def __call__(self, instant):
        if instant > len(self.timeline):
            for t in range(len(self.timeline), instant):
                entry = []
                for i in range(self.width):
                    entry.append(self.timeline[t - i - 1])
                self._forward_propagate(entry)
                self.timeline.append(*self.outputs[-1][1:])
            return self.timeline[-1]
        
        return self.timeline[instant - 1]
        
    @property
    def max_time(self):
        return len(self.desired) + self.width
    
    @property
    def width(self):
        return self.specs[0]
    
    def _prepare_training_set(self, samples):
        self.timeline = samples
        self.timeline = list(self.timeline)
        training = []
        desired = []
        for t in range(self.width + 1, len(samples)):
            training.append([])
            j = t - self.width - 1
            for i in range(self.width):
                training[j].append(samples[t - i - 1])
            desired.append([-1, *samples[t]])
        self.samples = training
        self.desired = np.array(desired)
        
        return (training, desired)
    
    def train(self, *args, **kwargs):
        if 'series' in kwargs:
            sets = self._prepare_training_set(kwargs['series'])
        MLP.train(self, samples=sets[0], desired=sets[1])