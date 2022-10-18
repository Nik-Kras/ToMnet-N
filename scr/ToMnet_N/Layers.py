#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class NeuralNetLayers:

The parent class for both the character net and the prediction net.
@author: Chuang, Yun-Shiuan
"""

import tensorflow as tf
from tensorflow import keras

from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras import activations

class CustomCnn(keras.layers.Layer):
    def __init__(self, input_tensor=None, activation="linear", filters=64,
                 UseTimeWrapper=False, regularisation_value = 0.001, **kwargs):
        super(CustomCnn, self).__init__(**kwargs)
        self.UseTimeWrapper = UseTimeWrapper
        self.input_tensor = input_tensor
        self.activation = activation
        self.filters = filters
        self.regularisation_value = regularisation_value

        if input_tensor is None:
            self.conv = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                activation=activation,
                                                padding="same",
                                                kernel_regularizer = keras.regularizers.l2(self.regularisation_value),
                                                bias_regularizer = keras.regularizers.l2(self.regularisation_value),
                                                kernel_initializer = tf.keras.initializers.HeNormal())
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                activation=activation,
                                                padding="same",
                                                input_shape=input_tensor,
                                                kernel_regularizer = keras.regularizers.l2(self.regularisation_value),
                                                bias_regularizer = keras.regularizers.l2(self.regularisation_value),
                                                kernel_initializer = tf.keras.initializers.HeNormal())
        if UseTimeWrapper: self.conv_handler = tf.keras.layers.TimeDistributed(self.conv)

    def call(self, inputs):
        if self.UseTimeWrapper: x = self.conv_handler(inputs)
        else: x = self.conv(inputs)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "regularisation_value": self.regularisation_value,
            "UseTimeWrapper": self.UseTimeWrapper,
            "input_tensor": self.input_tensor,
            "activation": self.activation,
            "filters": self.filters,
        })
        return config

def CustomCnnCharNet(input_tensor=None, activation="linear", filters=64, **kwargs):
    return CustomCnn(input_tensor=input_tensor, activation=activation, filters=filters, UseTimeWrapper=True, **kwargs)

def CustomCnnPredNet(input_tensor=None, activation="linear", filters=64, **kwargs):
    return CustomCnn(input_tensor=input_tensor, activation=activation, filters=filters, UseTimeWrapper=False, **kwargs)

class ResBlock(keras.layers.Layer):
    def __init__(self, UseTimeWrapper=False, filters=64, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu_conv = tf.keras.layers.Activation('relu')
        self.UseTimeWrapper = UseTimeWrapper
        if self.UseTimeWrapper:
            self.conv1 = CustomCnnCharNet(activation="linear", filters=filters)
            self.conv2 = CustomCnnCharNet(activation="linear", filters=filters)
        else:
            self.conv1 = CustomCnnPredNet(activation="linear", filters=filters)
            self.conv2 = CustomCnnPredNet(activation="linear", filters=filters)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu_conv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x + inputs)
        return x

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          "UseTimeWrapper": self.UseTimeWrapper
      })
      return config

def ResBlockCharNet(filters=64):
    return ResBlock(UseTimeWrapper=True, filters=filters)

def ResBlockPredNet(filters=64):
    return ResBlock(UseTimeWrapper=False, filters=filters)

class CustomLSTM(keras.layers.Layer):
    def __init__(self, num_hidden = 128,  **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)  # including name = name
        self.num_hidden = num_hidden
        self.lstm = LSTM(units=num_hidden,
                        activation = activations.tanh,
                        recurrent_activation = activations.sigmoid)
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = self.lstm(inputs)
        # x = self.bn(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_hidden": self.num_hidden
        })
        return config