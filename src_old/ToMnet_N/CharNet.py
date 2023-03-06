#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class CharNet(nnl.NeuralNetLayers):

For the single trajectory τi in the past episode, the
ToMnet forms the character embedding echar,i as follows. We
 (1) pre-process the data from each time-step by spatialising the actions,
 a(obs), concatenating these with the respective states, x(obs),
 (2) passing through a 5-layer resnet, with 32 channels, ReLU nonlinearities,
 and batch-norm, followed by average pooling.
 (3) We pass the results through an LSTM with 64 channels,
 with a linear output to either a 2-dim or 8-dim echar,i (no substantial difference in results).
@author: Chuang, Yun-Shiuan; Edwinn
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

from ToMnet_N.Layers import * # CustomCnnCharNet, ResBlockCharNet, CustomLSTM


# --------------------------------------------------------------
# CharNet is a layer, as it doesn't have separate and own training,
# it is simply a part of whole network, so can be considered as a layer
# --------------------------------------------------------------
class CharNet(keras.layers.Layer):

    def __init__(self, input_tensor, n, N_echar, filters=64):
        super(CharNet, self).__init__()

        # self.input_tensor = input_tensor
        self.n = n
        self.N_echar = N_echar

        self.conv = CustomCnnCharNet(input_tensor=input_tensor, filters=filters)
        self.res_blocks = [None] * n
        for i in range(n):
            self.res_blocks[i] = ResBlockCharNet(filters=filters)
        # Global Pool
        self.lstm = CustomLSTM()
        self.e_char = Dense(N_echar)

    def call(self, inputs):
        """
        Build the character net.
        """

        # input_tensor = self.input_tensor
        n = self.n
        N_echar = self.N_echar

        batch_size, trajectory_size, height, width, depth = inputs.get_shape().as_list()

        # --------------------------------------------------------------
        # Paper codes
        # (16, 10, 12, 12, 11) -> (16, 10, 12, 12, 32)
        # Add initial Conv2D layer
        # Conv2D standard: Shape = (batch_size, width, height, channels)
        # Conv2D takes only width x height x channels (12, 12, 11)
        # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
        # That process is happening in parallel for 16 objects in one batch
        # --------------------------------------------------------------
        x = self.conv(inputs)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 10, 12, 12, 11) -> (16, 10, 12, 12, 32)
        # Add n residual layers
        # Conv2D takes only width x height x channels (12, 12, 11)
        # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
        # That process is happening in parallel for 16 objects in one batch
        # --------------------------------------------------------------
        for i in range(n):
            x = self.res_blocks[i](x)  ### Possible error here!!!

        # --------------------------------------------------------------
        # Paper codes
        # (16, 10, 12, 12, 32) ->  (16, 10, 32)
        # Add average pooling
        # Collapse the spatial dimensions
        # --------------------------------------------------------------
        x = tf.reduce_mean(input_tensor=x, axis=[2, 3])

        # --------------------------------------------------------------
        # Paper codes
        # (16, 10, 32) ->  (16, 64)
        # Add LSTM
        # Standard: Shape = (batch_size, time_step, features)
        # for each x_i(t)(example_i's step_t): a (64, 1) = W(64, 32) * x (32, 1)
        # --------------------------------------------------------------
        x = self.lstm(x)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 64) -> (16, 4)
        # Add Fully connected layer
        # (batch_size, features) - > (batch_size, e_char)
        # --------------------------------------------------------------
        x = self.e_char(x)

        return x
