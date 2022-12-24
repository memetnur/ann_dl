#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:24:57 2020

@author: scli
"""


# =============================================================================
# SIMPLE TIME SERIES FORECASTING EXAMPLES
# =============================================================================

import numpy as np
# import tensorflow as tf
from tensorflow import keras
import pandas as pd
# import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt


# Note
# - size of input_time_steps
# - number of units in LSTM
# - difference between SimpleRNN and LSTM
# - it is hard to learn symbolic math
# - dense layer on top of LSTM layer
# - how to prepare dataset for training in time series case
# - use of relu 

	
# Univariate data preparation

# define a sequence 0-99
raw_seq = np.arange(100).astype(np.float32)
print(raw_seq)
time_index = np.arange(1,101)

df = pd.DataFrame(raw_seq, index=time_index, columns=['input series'])
print(df) 

# This function prepares the data: according to the size of input_time_steps, 
# batches or samples of size input_time_steps will be prepared. Also the values of 
# target time series y will be shifted by input_time_steps.
# For example, with data array of size 80 and input_time_steps=2, there will be 
# 98 batches or samples with size 2.

def setup_dataset(X, y, input_time_steps):
    Xs, ys = [], []
    for i in range(len(X) - input_time_steps):
        v = X.iloc[i:(i + input_time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + input_time_steps])
    return np.array(Xs), np.array(ys)


train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

# Note we need at least two input time steps to learn the task!
# Choose a number of time steps

input_time_steps = 1

X_train, y_train = setup_dataset(train, train, input_time_steps)
X_test, y_test = setup_dataset(test, test, input_time_steps)

print(X_train.shape, y_train.shape)

for i in range(len(X_train)):
 	print(X_train[i], y_train[i])

    
# =============================================================================
# MODEL
# =============================================================================


n_features = 1

# Define model
# Relu activation is used to output values in range [0,inf]

model = keras.Sequential()
#model.add(keras.layers.LSTM(64, activation='relu', input_shape=(input_time_steps, n_features)))
model.add(keras.layers.SimpleRNN(1, activation='relu', return_sequences=False, input_shape=(input_time_steps, n_features)))
model.add(keras.layers.Dense(1)) #default activation : linear

model.compile(optimizer='adam', loss='mse')
model.summary()

#y_train_pred = model.predict(X_train)

history = model.fit(X_train, y_train, epochs=800, validation_data=(X_test, y_test), verbose=1)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
 	print(y_pred[i], y_test[i])

# Single value prediction
# print(model.predict(np.array([100,101]).reshape(1,2,1).astype(np.float32)))

  
    




