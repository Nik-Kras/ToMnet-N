#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class PredNet(nnl.NeuralNetLayers):

In this and subsequent experiments,
we make three predictions:
  (1)next-step action,
  (2)which objects are consumed by the end of the episode, and
  (3) successor representations.
  We use a shared torso for these predictions, from which separate heads branch off.

  For the prediction torso, we
    (1) spatialise echar,i,
    (2) and concatenate with the query state;
    (3) this is passed into a 5-layer resnet, with 32 channels, ReLU nonlinearities, and batch-norm.

  Consumption prediction head.
  From the torso output:
    （1) a 1-layer convnet with 32 channels and ReLUs, followed by average pooling, and
     (2) a fully-connected layer to 4-dims,
     (3) followed by a sigmoid. This gives the respective Bernoulli probabilities
     that each of the four objects will be consumed by the end of the episode.
     [Unlike the paper, I replaced this sigmoid unit by a softmax unit.]
@author: Chuang, Yun-Shiuan
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

from ToMnet_N.Layers import * # CustomCnnCharNet, ResBlockCharNet, CustomLSTM

# --------------------------------------------------------------
# CharNet is a layer, as it doesn't have separate and own training,
# it is simply a part of whole network, so can be considered as a layer
# --------------------------------------------------------------
class PredNet(keras.layers.Layer):

    def __init__(self, n):
        super(PredNet, self).__init__()
        self.n = n

        self.e_char_shape = 8
        self.current_state_shape = (12, 12, 6)

        self.conv_1 = CustomCnnPredNet(input_tensor=self.current_state_shape)
        self.res_blocks = [None] * n
        for i in range(n):
          self.res_blocks[i] = ResBlockPredNet()
        self.conv_2 = CustomCnnPredNet(activation='relu')
        self.fc = Dense(units=60, activation=activations.relu)
        # drop_out_1 = Dropout(rate = 0.2) ### Could be added in the future
        self.goal_predict = Dense(units=4, activation=activations.linear)

    def call(self, inputs):
        """
        Build the character net.
        """
        ### Check that inputs.shape == (None, 13, 12, 8)

        # Get shapes
        # batch_size, height, width, depth = inputs.get_shape().as_list()
        # _, embedding_length = e_char.get_shape().as_list()
        n = self.n

        # --------------------------------------------------------------
        # Paper codes
        # (16, 13, 12, 8) -> (16, 12, 12, 6) + (16, 8)
        # Decompose input data
        # Initially in is a mix of Current State and e_char embedding space
        # --------------------------------------------------------------
        input_current_state = inputs[..., 0:12, 0:12, 0:6]
        e_char = inputs[..., 12, 0, :]

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 6) -> (16, 12, 12, 32)
        # Use 3x3 conv layer to shape the depth to 32
        # to enable resnet to work (addition between main path and residual connection)
        # --------------------------------------------------------------
        x = self.conv_1(input_current_state)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 12, 12, 32)
        # Add n residual layers
        # Conv2D takes only width x height x channels (12, 12, 11)
        # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
        # That process is happening in parallel for 16 objects in one batch
        # --------------------------------------------------------------
        for i in range(n):
          x = self.res_blocks[i](x)    ### Possible error here!!!

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 12, 12, 32)
        # Add CNN after Res Blocks
        # --------------------------------------------------------------
        x = self.conv_2(x)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 32)
        # Add average pooling
        # Collapse the spatial dimensions
        # --------------------------------------------------------------
        x = tf.reduce_mean(input_tensor=x, axis=[1, 2])

        # --------------------------------------------------------------
        # Paper codes
        # (16, 32) + (16, 8) -> (16, 32, 1) + (16, 8, 1) - >
        # (16, 40, 1) -> (16, 40)
        # Concatenate tensor with e_char
        # Concatenation requires a common dimentions which cannot be a batch
        # --------------------------------------------------------------
        x = tf.expand_dims(x, axis=-1)
        e_char = tf.expand_dims(e_char, axis=-1)

        x = tf.keras.layers.Concatenate(axis=1)([x, e_char])
        x = x[..., 0]

        # --------------------------------------------------------------
        # Paper codes
        # (16, 40) -> (16, 60) -> (16, 4)
        # Fully connected layer with dropout for regularization
        # --------------------------------------------------------------
        x = self.fc(x)
        x =  self.goal_predict(x)

        return x