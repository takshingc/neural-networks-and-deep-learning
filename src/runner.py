#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:43:19 2019

@author: terrence
"""

import mnist_loader
import tsnetwork
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = np.array(list(training_data))
test_data = np.array(list(test_data))

if __name__=='__main__':
    net = tsnetwork.Network([784, 30, 10])
    net.SGD(training_data[:,0], training_data[:,1], 30, 10, 3.0, test_data[:,0], test_data[:,1])