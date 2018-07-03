# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:22:16 2018

@author: luiz_
"""

try:
    from hopfield import Hopfield
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(".."))
    from hopfield import Hopfield
    
import matplotlib.pyplot as plt
import numpy as np
import random

def make_image(x):
    a = x.reshape(9, 5)
    im = plt.imshow(a, cmap='inferno')
    plt.show()
    return im

def scale_image(data):
    new_data = np.zeros(np.array(data.shape) * 10)

    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * 10: (j+1) * 10, k * 10: (k+1) * 10] = data[j, k]
    return new_data

def catimages(i1, i2, i3):
    i11 = i1.reshape(9, 5)
    i22 = i2.reshape(9, 5)
    i33 = i3.reshape(9, 5)
    im = i11
    null_col = np.zeros([i11.shape[0], 1]) - 1
    im = np.concatenate((im, null_col), axis=1)
    im = np.concatenate((im, i22), axis=1)
    im = np.concatenate((im, null_col), axis=1)
    im = np.concatenate((im, i33), axis=1)
    return im

def make_one():
    # matrix for number one
    one = np.zeros([9, 5]) - 1
    one[:, 3] = np.ones(9)
    one[:, 2] = np.ones(9)
    one[1, 1] = 1
    return one.reshape(1, 45)

def make_two():
    # for number two
    two = np.ones([9, 5])
    for i in range(3):
        for j in range(2, 4):
            two[j, i] = -1
    two[5:, :] = np.flipud(np.fliplr(two[:4, :]))
    return two.reshape(1, 45)

def make_three():
    # for number three
    three = np.ones([9, 5])
    for i in range(3):
        for j in range(2, 4):
            three[j, i] = -1
    three[5:, :] = np.flipud(three[:4, :])
    return three.reshape(1, 45)

def make_four():
    # number four
    four = np.ones([9, 5])
    four[:3, 2] = np.zeros([3]) - 1
    four[5:, :3] = np.zeros([4, 3]) - 1
    return four.reshape(1, 45)

def make_number(n):
    if n == 1:
        return make_one()
    if n == 2:
        return make_two()
    if n == 3:
        return make_three()
    if n == 4:
        return make_four()
    else:
        raise NotImplementedError('NÃO FIZ')

def add_noise(x, prob=.2):
    for i in range(len(x[0])):
        r = random.uniform(0, 1)
        if r < prob:
            x[0][i] = -1 if x[0][i] == 1 else 1

if __name__ == "__main__":
    net = Hopfield(45, 4)
    patterns = [make_number(i) for i in range(1, 5)]
    net.define_weights(patterns)
    
    noisy = []
    for pattern in patterns:
        for i in range(3):
            p = np.array([x for x in pattern])
            add_noise(p)
            noisy.append(p)
    
    for i in range(len(noisy)):
        im = catimages(patterns[int(np.floor(i / 3))], noisy[i], net(noisy[i]))
        plt.imshow(im, cmap='inferno')
        im = scale_image(im)
        plt.imsave('im%s.png' % i, im, cmap='inferno')
        plt.show()