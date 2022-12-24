#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:06:57 2020

@author: scli
"""


# =============================================================================
# WAEDENSWIL WEATHER FORECAST
# =============================================================================


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data and make dataframe
column_names = ['day','month','year', 'mint','mxt','p']
df = pd.read_csv('WAE_Weather.csv', usecols=column_names, sep=';')
# Show some data
df.head()

# Make single column date
date = pd.to_datetime(df.year*10000+df.month*100+df.day,format='%Y%m%d')
df.insert (0, "date", date)
df = df.drop(['day', 'month','year'], axis=1)
# Show some data
df.head()


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

TRAIN_SPLIT = 10000

# Setting seed to ensure reproducibility.
tf.random.set_seed(10)


# =============================================================================
# SETUP DATA FOR UNIVARIATE MODEL
# =============================================================================

# Let's first extract only the max temperature from the dataset.

uni_data = df['mxt']
uni_data.index = df['date']
uni_data.head()

# Plot
uni_data.plot(subplots=True)
# Show data
uni_data = uni_data.values
print(uni_data)

# Normalize data * Note: The mean and standard deviation should only be computed using the training data.
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

# Set up data: the model will be given the last 20 recorded temperature observations, 
# and needs to learn to predict the temperature at the next time step.
univariate_past_history = 20
univariate_future_target = 1

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])

# The information given to the network is given in blue, and it must predict 
# the value at the red cross.

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

# =============================================================================
# BASE MODEL/BENCHMARK
# =============================================================================

def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')

# =============================================================================
# LSTM MODEL
# =============================================================================

BATCH_SIZE = 128
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# Model
simple_gru_model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_gru_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(simple_gru_model.predict(x).shape)

# val_univariate.take(1)

# Due to the large size of the dataset, in the interest of saving time, 
# each epoch will only run for 200 steps, instead of the complete training data 
# as normally done.

EVALUATION_INTERVAL = 200
EPOCHS = 10

history = simple_gru_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

plot_train_history(history,
                   'Training and validation loss')

# Predict using the simple LSTM model
for x, y in val_univariate.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_gru_model.predict(x)[0]], 0, 'Simple GRU model')
  plot.show()


