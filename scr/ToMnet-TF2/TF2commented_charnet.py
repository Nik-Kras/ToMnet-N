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
#import numpy as np
#from tensorflow.contrib import rnn
import TF2commented_nn_layers as nnl

from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras import activations


# For debugging
import pdb
class CharNet(nnl.NeuralNetLayers):

  def __init__(self):

      # hyperparameter for batch_normalization_layer()
      self.BN_EPSILON = 0.001

      # hyperparameter for create_variables()
      # this is especially for "tf.contrib.layers.l2_regularizer"
#      self.WEIGHT_DECAY = 0.00002
      self.WEIGHT_DECAY = 0.001
      self.MAZE_DEPTH = 11

  def build_charnet(self,input_tensor, n, num_classes, reuse, train, keras_model = False):
    '''
    Build the character net.

    if keras_model = True
      :param input_tensor:
      :param n: the number of layers in the resnet
      :param num_classes:
      :param reuse: ?
      :param train: If training, there will be dropout in the LSTM. For validation/testing,
        droupout won't be applied.
      :return model: full Char Net model able to learn and be called

    if keras_model = False
      :param input_tensor:
      :param n: the number of layers in the resnet
      :param num_classes:
      :param reuse: ?
      :param train: If training, there will be dropout in the LSTM. For validation/testing,
        droupout won't be applied.
      :return layers[-1]: "logits" is the output of the charnet (including ResNET and LSTM) and is the input for a softmax layer
    '''
    if keras_model:

      input_shape = batch_size, trajectory_size, height, width, depth  = input_tensor.get_shape().as_list()

      # --------------------------------------------------------------
      # Paper codes
      # (16, 10, 12, 12, 11) -> (16, 10, 12, 12, 32)
      # Add initial Conv2D layer
      # Conv2D standard: Shape = (batch_size, width, height, channels)
      # Conv2D takes only width x height x channels (12, 12, 11)
      # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
      # That process is happening in parallel for 16 objects in one batch
      # --------------------------------------------------------------
      inputs = tf.keras.Input(shape=(trajectory_size, height, width, depth), batch_size=batch_size)
      conv_2d_layer = tf.keras.layers.Conv2D(filters=32,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             # activation='relu',
                                             padding="same",
                                             input_shape=(height, width, depth))  # kernel_size=(3, 3) ? (11,11) ? If an input image has 3 channels (e.g. a depth of 3), then a filter applied to that image must also have 3 channels (e.g. a depth of 3). In this case, a 3×3 filter would in fact be 3x3x3
      conv_2d_handler = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)

      # --------------------------------------------------------------
      # Paper codes
      # (16, 10, 12, 12, 11) -> (16, 10, 12, 12, 32)
      # Add n residual layers
      # Conv2D takes only width x height x channels (12, 12, 11)
      # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
      # That process is happening in parallel for 16 objects in one batch
      # --------------------------------------------------------------
      res_layers = []
      prev_layer = conv_2d_handler
      for i in range(n):
        # Conv2D to process 12x12x11 time-frame
        conv_layer = tf.keras.layers.Conv2D(filters=32,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             input_shape=(height, width, depth))
        # Thing that will pass 10 time frames iteratively to Conv2D. Outputs 10x12x12x11
        conv_layer_handler = tf.keras.layers.TimeDistributed(conv_layer)(prev_layer)

        # NIKITA: ORIGINALLY I MUST INCLUDE BATCH NORMALISATION HERE, BUT I DON'T SEE SENSE IN IT
        # Normalize the whole batch
        # mean, variance = tf.nn.moments(x=input_layer, axes=[0, 1, 2])
        # batch_norm = BatchNormalization()
        # bn_layer = self.batch_normalization_layer(conv_layer, out_channel)
        ### res_batch_1 = BatchNormalization(axes=[2, 3, 4])(conv_layer_handler)  # If want to add Normalisation - Use this!
        res_conv_1 = tf.nn.relu(conv_layer_handler)

        # Conv2D to process 12x12x11 time-frame
        conv_layer_2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            input_shape=(height, width, depth))
        # Thing that will pass 10 time frames iteratively to Conv2D. Outputs 10x12x12x11
        conv_layer_handler_2 = tf.keras.layers.TimeDistributed(conv_layer_2)(res_conv_1)
        res_conv_2 = conv_layer_handler_2

        res_layers[i] = tf.nn.relu(res_conv_2 + prev_layer)
        prev_layer = res_layers[i]

      # --------------------------------------------------------------
      # Paper codes
      # (16, 10, 12, 12, 32) ->  (16, 10, 32)
      # Add average pooling
      # Collapse the spatial dimensions
      # --------------------------------------------------------------
      global_pool = tf.reduce_mean(input_tensor=prev_layer, axis=[2, 3])

      # --------------------------------------------------------------
      # Paper codes
      # (16, 10, 32) ->  (16, 64)
      # Add LSTM
      # Standard: Shape = (batch_size, time_step, features)
      # for each x_i(t)(example_i's step_t): a (64, 1) = W(64, 32) * x (32, 1)
      # --------------------------------------------------------------
      num_hidden = 64
      output_keep_prob = 0.2  # Regularization during training
      lstm_layer = LSTM(units=num_hidden,
                        activation = activations.tanh,
                        recurrent_activation = activations.sigmoid,
                        recurrent_dropout = output_keep_prob,
                        dropout = output_keep_prob)(global_pool)
      lstm_batch_norm = BatchNormalization()(lstm_layer)

      # --------------------------------------------------------------
      # Paper codes
      # (16, 64) -> (16, 4)
      # Add Fully connected layer
      # (batch_size, features) - > (batch_size, e_char)
      # --------------------------------------------------------------
      N_echar = num_classes
      e_char = Dense(N_echar)(lstm_batch_norm)

      model = Model(inputs=inputs, outputs=e_char)
      # How should I compile it and calculate the loss?
      # model.compile(optimizer='adam',
      #               loss='categorical_crossentropy',
      #               metrics=['accuracy'])
      return model

    else:
      # pdb.set_trace()
      layers = []

      # --------------------------------------------------------------
      # Paper codes
      # Regard each step independently in the resnet
      # (16, 10, 12, 12, 11) -> (160, 12, 12, 11)
      # input_tensor.shape = (16, 10, 12, 12, 11)
      # 16: 16 trajectories
      # 10: each trajectory has 10 steps
      # 12, 12, 11: maze height, width, depth
      # --------------------------------------------------------------
      # pdb.set_trace()
      layers.append(input_tensor)
      # pdb.set_trace()
      batch_size, trajectory_size, height, width, depth  = layers[-1].get_shape().as_list()
      # layers[-1] = input_tensor = (16, 10, 12, 12, 11)
      step_wise_input = tf.reshape(layers[-1], [batch_size * trajectory_size, height, width, depth])
      # resnet_iput = (160, 12, 12, 11)
      layers.append(step_wise_input)

      # --------------------------------------------------------------
      # Paper codes
      # (160, 12, 12, 11) -> (160, 12, 12, 32)
      # Use 3x3 conv layer to shape the depth to 32
      # to enable resnet to work (addition between main path and residual connection)
      # --------------------------------------------------------------
      with tf.compat.v1.variable_scope('conv_before_resnet', reuse = reuse):
        #pdb.set_trace()
        conv_before_resnet = self.conv_layer_before_resnet(layers[-1])
        layers.append(conv_before_resnet)
        _, _, _, resnet_input_channels  = layers[-1].get_shape().as_list()


      #Add n residual layers
      for i in range(n):
        with tf.compat.v1.variable_scope('conv_%d' %i, reuse=reuse):

            # --------------------------------------------------------------
            # Paper codes
            # (160, 12, 12, 32) -> (160, 12, 12, 32)
            # layers[-1] = intput_tensor = (16, 10, 12, 12, 32)
            # 160: 160 steps (16 trajectories x 10 steps/trajectory)
            # 10: each trajectory has 10 steps
            # 12, 12, 11: maze height, width, depth

            # block = (160, 12, 12, 32)
            # 160: 160 steps (16 trajectories x 10 steps/trajectory)
            # 12, 12, 32: maze height, width, output channels (as in the paper)
            # --------------------------------------------------------------

            #pdb.set_trace()
            # layers[-1] = (16, 10, 12, 12, 11)
            resnet_input = layers[-1]
            # resnet_input = (160, 12, 12, 11)

            block = self.residual_block(resnet_input, resnet_input_channels)
            self.activation_summary(block)
            layers.append(block)

      # --------------------------------------------------------------
      #Add average pooling
      # Paper codes
      # (160, 12, 12, 32) ->  (160, 32)
      # # collapse the spacial dimension
      #
      # layers[-1] = block = (160, 12, 12, 11)
      # 160: 160 steps (16 trajectories x 10 steps/trajectory)
      # 12, 12, 32: maze height, width, output channels (32 as in the paper)
      #
      # avg_pool = (160, 32)
      # 160: 160 steps (16 trajectories x 10 steps/trajectory)
      # 32: output channels
      # --------------------------------------------------------------
      with tf.compat.v1.variable_scope('average_pooling', reuse=reuse):
        avg_pool = self.average_pooling_layer(block)
        layers.append(avg_pool)



      #Add LSTM layer
      # pdb.set_trace()
      with tf.compat.v1.variable_scope('LSTM', reuse=reuse):

        # --------------------------------------------------------------
        # Paper codes
        # (160, 32) ->  (16, 4)
        #
        # avg_pool = (160, 32)
        # 160: 160 steps (16 trajectories x 10 steps/trajectory)
        # 32: output channels

        # lstm = (16, 4)
        # 16: batch_size (16 trajectories)
        # 4: num_classes
        # --------------------------------------------------------------

        # --------------------------------------------------------------

        # layers[-1] = avg_pool = (160, 32)
        _, resnet_output_channels = layers[-1].get_shape().as_list()

        # layers[-1] = avg_pool = (160, 32)
        lstm_input = tf.reshape(layers[-1], [batch_size, trajectory_size, resnet_output_channels])
        # lstm_input = (16, 10, 32)

        # lstm_input = (16, 10, 32)
        lstm = self.lstm_layer(lstm_input, train, num_classes)
        # lstm = (16, 64)

        layers.append(lstm)

      # --------------------------------------------------------------
      #Fully connected layer
      # Paper codes
      # (16, 64) -> (16, 4)
      # def output_layer(self,input_layer, num_labels):
      #   '''
      #   A linear layer.
      #   :param input_layer: 2D tensor
      #   :param num_labels: int. How many output labels in total?
      #   :return: output layer Y = WX + B
      #   '''
      # --------------------------------------------------------------
      with tf.compat.v1.variable_scope('fc', reuse=reuse):
        # layers[-1] = (16, 64)
        output = self.output_layer(layers[-1], num_classes)
        # output = (16, 4)
        layers.append(output)
      return layers[-1]
