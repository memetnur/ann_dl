#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:40:07 2020

@author: scli
"""

# =============================================================================
# MULTI-PERCEPTRON
# =============================================================================

import numpy as np

# INPUT/TARGET EXAMPLES ------------------------------------------------------------
# input
X = np.array( ((1,1,1,1,1), (1,0,1,0,1), (0,0,1,0,0), (0,0,0,0,0)) )
# target
T = np.array(((1,1,1),(1,0,1),(0,1,0),(0,0,0)))

# INITIALIZE WEIGHTS/BIAS BY ZEROS
W_init = np.zeros((T.shape[1],X.shape[1])) #this gives a 5 (input size) times 3 (output size) matrix of zeros
b_init = 0

# EXTENDED INPUT ----------------------------------------------------------
X_ext = np.append(X, [[1],[1],[1],[1]], 1)
W = np.append(W_init, [[0],[0],[0]], 1)

# PERCEPTRON LEARNING -------------------------------------------------------
epochs = 7 #number of epochs

X_ext_t = X_ext.transpose() #transpose matrix for matrix multiplication

for n in range(epochs):
    for i in range(len(X_ext)):
        y_i = np.where(np.dot(W,X_ext_t[:,i]) >= 0, 1, 0)
        W = W + np.outer((T[i]-y_i),X_ext_t[:,i]) #compute outer product to update weight matrix
        
y = np.where(np.dot(W,X_ext_t) >= 0, 1, 0) 

# OUTPUT
print(y.transpose()) #print output of perceptron learning
print(T) #print target
print(W) #print weights






