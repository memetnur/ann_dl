#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:12:10 2020

@author: scli
"""
# =============================================================================
# PERCEPTRON LEARNING RULE
# =============================================================================

# LEARN LOGICAL AND OPERATOR WITH HARDLIM/THRESHOLD PERCEPTRON --------------------------

import numpy as np

# INPUT/TARGET ------------------------------------------------------------
# input
X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )
# target
T = np.array([1,0,0,0]) #supervised learning we have to specifize the output = target

# INITIAL WEIGHTS/BIAS -------------------------------------------
W_init = np.array([0,0])
b_init = 0

# EXTENDED INPUT ----------------------------------------------------------
X_ext = np.hstack((X,np.ones((4,1))))
W = np.append(W_init,b_init)

# PERCEPTRON LEARNING -------------------------------------------------------
epochs = 7 #number of epochs

for n in range(epochs):
    for i in range(len(X_ext)):
        y_i = np.where(np.dot(X_ext[i,:],W) >= 0, 1, 0)
        W = W + (T[i]-y_i)*X_ext[i,:]
        
y = np.where(np.dot(X_ext,W) >= 0, 1, 0)   

print(y)
print(W)

 