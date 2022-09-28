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
    def __init__(self, input_tensor=None, activation="linear", filters=32, UseTimeWrapper=False):
        super(CustomCnn, self).__init__()
        self.input_tensor = input_tensor
        self.activation = activation
        self.filters = filters
        if input_tensor is None:
          self.conv = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  activation=activation,
                                  padding="same")
        else:
          self.conv = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  activation=activation,
                                  padding="same",
                                  input_shape=input_tensor)
        if UseTimeWrapper: self.conv_handler = tf.keras.layers.TimeDistributed(self.conv)
        self.UseTimeWrapper = UseTimeWrapper


    def call(self, inputs):
      if self.UseTimeWrapper: x = self.conv_handler(inputs)
      else: x = self.conv(inputs)

      return x

    def get_config(self):
      config = super().get_config()
      config.update({
          "input_tensor": self.input_tensor,
          "activation": self.activation,
          "filters": self.filters,
          "UseTimeWrapper": self.UseTimeWrapper
      })
      return config

def CustomCnnCharNet(input_tensor=None, activation="linear", filters=32):
    return CustomCnn(input_tensor=input_tensor, activation=activation, filters=filters, UseTimeWrapper=True)

def CustomCnnPredNet(input_tensor=None, activation="linear", filters=32):
    return CustomCnn(input_tensor=input_tensor, activation=activation, filters=filters, UseTimeWrapper=False)

class ResBlock(keras.layers.Layer):
    def __init__(self, UseTimeWrapper=False):
        super(ResBlock, self).__init__()
        if UseTimeWrapper:
            self.conv1 = CustomCnnCharNet(activation="relu")
            # Use Batch Normalisation and then Relu activation in future!
            self.conv2 = CustomCnnCharNet(activation="linear")
            # Use Batch Normalisation in future!
        else:
            self.conv1 = CustomCnnPredNet(activation="relu")
            # Use Batch Normalisation and then Relu activation in future!
            self.conv2 = CustomCnnPredNet(activation="linear")
            # Use Batch Normalisation in future!
        self.UseTimeWrapper = UseTimeWrapper

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.nn.relu(x + inputs)
        return x

    def get_config(self):
      config = super().get_config()
      config.update({
          "UseTimeWrapper": self.UseTimeWrapper
      })
      return config

def ResBlockCharNet():
    return ResBlock(UseTimeWrapper=True)

def ResBlockPredNet():
    return ResBlock(UseTimeWrapper=False)

class CustomLSTM(keras.layers.Layer):
  def __init__(self, num_hidden = 64, output_keep_prob = 0.2):
    super(CustomLSTM, self).__init__()
    self.num_hidden = num_hidden
    self.output_keep_prob = output_keep_prob
    self.lstm = LSTM(units=num_hidden,
                    activation = activations.tanh,
                    recurrent_activation = activations.sigmoid,
                    recurrent_dropout = output_keep_prob,
                    dropout = output_keep_prob)
    self.bn = BatchNormalization()

  def call(self, inputs):
    x = self.lstm(inputs)
    x = self.bn(x)
    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        "output_keep_prob": self.output_keep_prob,
        "num_hidden": self.num_hidden
    })
    return config