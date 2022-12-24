#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:09:09 2020

@author: scli
"""

# =============================================================================
# SIMPLE RNN DEMONSTRATING EXAMPLE FROM LECTURE
# =============================================================================

import tensorflow as tf
import numpy as np

inputs = np.array([[[2,-0.5,1,1]]]).astype(np.float32).reshape(1,4,1) 
# inputs: A 3D tensor, with shape [batch, timesteps, feature]
print(inputs)

# model
simple_rnn = tf.keras.layers.SimpleRNN(1, activation=None, use_bias=False, return_sequences=True, return_state=True)
# return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. 
# return_state: Boolean. Whether to return the last state in addition to the output. Default: False

# build model by calling input
simple_rnn(inputs)
# set weights
w1=np.array(1).reshape(1,1).astype(np.float32) # input's weight
w2=np.array(1).reshape(1,1).astype(np.float32) # hidden state's weight
simple_rnn.set_weights([w1,w2])
# show the weights
simple_rnn.get_weights()

# output
whole_sequence_output, final_state = simple_rnn(inputs)
print(whole_sequence_output, final_state)




