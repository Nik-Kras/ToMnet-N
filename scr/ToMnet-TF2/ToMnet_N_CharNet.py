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

# from keras.models import Model
from keras.layers import Dense
# from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras import activations


# --------------------------------------------------------------
# CharNet is a layer, as it doesn't have separate and own training,
# it is simply a part of whole network, so can be considered as a layer
# --------------------------------------------------------------
class CharNet(keras.layers.Layer):

    def __init__(self, input_tensor, n, N_echar):
        super(CharNet, self).__init__()

        # self.input_tensor = input_tensor
        self.n = n
        self.N_echar = N_echar

        # hyperparameter for batch_normalization_layer()
        self.BN_EPSILON = 0.001

        # hyperparameter for create_variables()
        # this is especially for "tf.contrib.layers.l2_regularizer"
        #      self.WEIGHT_DECAY = 0.00002
        self.WEIGHT_DECAY = 0.001
        self.MAZE_DEPTH = 11

    def call(self, inputs):
        """
        Build the character net.
        """

        # input_tensor = self.input_tensor
        n = self.n
        N_echar = self.N_echar

        batch_size, trajectory_size, height, width, depth  = inputs.get_shape().as_list()

        # --------------------------------------------------------------
        # Paper codes
        # (16, 10, 12, 12, 11) -> (16, 10, 12, 12, 32)
        # Add initial Conv2D layer
        # Conv2D standard: Shape = (batch_size, width, height, channels)
        # Conv2D takes only width x height x channels (12, 12, 11)
        # Time Distributed layer feeds a Conv2D with time-frames (10 frames)
        # That process is happening in parallel for 16 objects in one batch
        # --------------------------------------------------------------
        # inputs = tf.keras.Input(shape=(trajectory_size, height, width, depth), batch_size=batch_size) # This is for model definition, not layer definition
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
        e_char = Dense(N_echar)(lstm_batch_norm)

        return e_char
