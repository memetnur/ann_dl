#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:52:17 2020

@author: scli
"""

## SOME BUG FIX
#import appnope
#appnope.nope()


# =============================================================================
# IMPLEMENTATION OF AN ARTIFICAL NEURON
# =============================================================================

# IMPLEMENTING THE AND OPERATION --------------------------

# PYTHONIC IMPLEMENTATION ------------------------------------------------------------
# input
X = [[1,1],
	[1,0],
	[0,1],
	[0,0]]

# weights and threshold
weights = [1,1]
threshold = -1.5

# preallocate output list
output = [0 for _ in range(len(X))]

# compute output
for i in range(len(X)):
    net = 0
    for j in range(len(weights)):
            net += weights[j] * X[i][j]
    output[i] = 1.0 if net >= -threshold else 0.0
        
print(output)   



# NUMPY IMPLEMENTATION------------------------------------------------------------

import numpy as np

# input
X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )

# weights and threshold
weights = np.array([1, 1]) 
threshold = -1.5  

# compute output
net = np.dot(X,weights)
output = np.where(net >= -threshold, 1, 0)
print(output)



# NUMPY IMPLEMENTATION AS FUNCTION ------------------------------------------------------------

import numpy as np

def artificial_neuron(X,weights,threshold):
    net = np.dot(X,weights)
    return np.where(net >= threshold, 1, 0)

# input
X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )

# weights and threshold
weights = np.array([0.3, 0.3]) 
threshold = 0.5 

output = artificial_neuron(X,weights,threshold)
print("output =", output)



#import time
# # PYTHONIC IMPLEMENTATION ------------------------------------------------------------
# # input
#
#X = [[1,1],
# 	[1,0],
# 	[0,1],
# 	[0,0]]
#
# # weights and threshold
#weights = [0.4,0.4]
#threshold = 0.5
#
# # preallocate output list
#output = [0 for _ in range(len(X))]
#
# # compute output
#start = time.time()
#for i in range(len(X)):
#    net = 0
#for j in range(len(weights)):
#net += weights[j] * X[i][j]
#output[i] = 1.0 if net >= threshold else 0.0
#
#print(output) 
#end = time.time()
#print(end - start)
# 
#
# # NUMPY IMPLEMENTATION------------------------------------------------------------
#
#import numpy as np
#
# # input
#X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )
#
# # weights and threshold
#weights = np.array([0.6, 0.3]) 
#threshold = 0.5  
#
#start = time.time()
## compute output
#net = np.dot(X,weights)
#output = np.where(net > threshold, 1, 0)
#print(output)    
#end = time.time()
#print(end - start)




    
    
    
    
    