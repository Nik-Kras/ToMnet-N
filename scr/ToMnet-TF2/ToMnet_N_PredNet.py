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

# from keras.models import Model
from keras.layers import Dense
# from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Dropout
from keras import activations
from keras.layers import Concatenate


# --------------------------------------------------------------
# CharNet is a layer, as it doesn't have separate and own training,
# it is simply a part of whole network, so can be considered as a layer
# --------------------------------------------------------------
class PredNet(keras.layers.Layer):

    def __init__(self, n):
        super(PredNet, self).__init__()
        self.n = n

    def call(self, inputs):
        """
        Build the character net.
        """

        n = self.n

        # --------------------------------------------------------------
        # Paper codes
        # Decompose input data
        # Initially in is a mix of Current State and e_char embedding space
        # --------------------------------------------------------------
        e_char = inputs["e_char"]
        current_state_tensor = inputs["input_current_state"]

        # --------------------------------------------------------------
        # Paper codes
        # Get the tensor size: (16, 12, 12, 6) <-> (batch size, height, width, channels)
        # --------------------------------------------------------------
        batch_size, height, width, depth = current_state_tensor.get_shape().as_list()

        # --------------------------------------------------------------
        # Paper codes
        # Get the character embedding size: (16, 8) <-> (batch size, e_char)
        # --------------------------------------------------------------
        _, embedding_length = e_char.get_shape().as_list()

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 6) -> (16, 12, 12, 32)
        # Use 3x3 conv layer to shape the depth to 32
        # to enable resnet to work (addition between main path and residual connection)
        # --------------------------------------------------------------
        conv_2d_layer = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               # activation='relu',
                               padding="same",
                               input_shape=(height, width, depth))(current_state_tensor)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 12, 12, 32)
        # Add n residual layers
        # Conv2D takes only width x height x channels (12, 12, 11)
        # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
        # That process is happening in parallel for 16 objects in one batch
        # --------------------------------------------------------------
        res_layers = []
        prev_layer = conv_2d_layer
        for i in range(n):

            conv_layer = tf.keras.layers.Conv2D(filters=32,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="same",
                                                input_shape=(height, width, depth))(prev_layer)

            ### res_batch_1 = BatchNormalization(axes=[2, 3, 4])(conv_layer)  # If want to add Normalisation - Use this!
            res_conv_1 = tf.nn.relu(conv_layer)

            conv_layer_2 = tf.keras.layers.Conv2D(filters=32,
                                                  kernel_size=(3, 3),
                                                  strides=(1, 1),
                                                  padding="same",
                                                  input_shape=(height, width, depth))(res_conv_1)

            res_conv_2 = conv_layer_2

            res_layers[i] = tf.nn.relu(res_conv_2 + prev_layer)
            prev_layer = res_layers[i]

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 12, 12, 32)
        # Add CNN after Res Blocks
        # --------------------------------------------------------------
        conv_2d_layer_after_res = tf.keras.layers.Conv2D(filters=32,
                                                       kernel_size=(3, 3),
                                                       strides=(1, 1),
                                                       activation='relu',
                                                       padding="same",
                                                       input_shape=(height, width, depth))(prev_layer)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 32) -> (16, 32)
        # Add average pooling
        # Collapse the spatial dimensions
        # --------------------------------------------------------------
        global_pool = tf.reduce_mean(input_tensor=conv_2d_layer_after_res, axis=[1, 2])

        # --------------------------------------------------------------
        # Paper codes
        # (16, 32) + (16, 8) -> (16, 40)
        # Concatenate tensor with e_char
        # --------------------------------------------------------------
        merge = Concatenate([global_pool, e_char])

        # --------------------------------------------------------------
        # Paper codes
        # (16, 40) -> (16, 60) -> (16, 4)
        # Fully connected layer with dropout for regularization
        # --------------------------------------------------------------
        dense_1 = Dense(units=60, activation=activations.relu)(merge)
        drop_out_1 = Dropout(rate = 0.2)(dense_1)
        output = Dense(units=60, activation=activations.linear)(drop_out_1)

        return output