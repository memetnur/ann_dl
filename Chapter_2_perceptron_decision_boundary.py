#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:21:09 2020

@author: scli
"""


# ## SOME BUG FIX
# import appnope
# appnope.nope()


# =============================================================================
# PERCEPTRON WITH DECISION BOUNDARY
# =============================================================================

# LEARN LOGICAL AND OPERATOR WITH HARDLIM/THRESHOLD PERCEPTRON --------------------------

import numpy as np
import matplotlib.pyplot as plt 

# INPUT/TARGET ------------------------------------------------------------
# input
X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )
# target
T = np.array([1,0,0,0])

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
        a = [0,-W[2]/W[1]] #compute decision boundary
        b = [-W[2]/W[0],0] #compute decision boundary
        #plt.figure()
        plt.scatter([1, 1, 0, 0], [1, 0, 1, 0], c=T)
        plt.plot(a,b)
        plt.pause(0.2)
        #plt.show()

y = np.where(np.dot(X_ext,W) >= 0, 1, 0)   

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()
        
y = np.where(np.dot(X_ext,W) >= 0, 1, 0)   

print(y)
print(W)

# final decision boundary
a = [0,-W[2]/W[1]] #compute decision boundary
b = [-W[2]/W[0],0] #compute decision boundary

plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(a,b)
plt.show()


import matplotlib.pyplot as plt
x=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
y=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
for i in range(len(x)):
    plt.figure()
    plt.plot(x[i],y[i])
    # Show/save figure as desired.
    plt.show()







    