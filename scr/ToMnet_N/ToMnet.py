import tensorflow as tf
from tensorflow import keras

import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras import activations

from ToMnet_N.CharNet import *
from ToMnet_N.PredNet import *

class ToMnet(Model):

    LENGTH_E_CHAR = 8
    NUM_RESIDUAL_BLOCKS = 8

    def __init__(self, ts, w, h, d, Ne_char=8, N_res_blocks=8, filters=32):
        super(ToMnet, self).__init__(name="ToMnet-N")

        self.MAX_TRAJECTORY_SIZE = ts  # 20-50
        self.MAZE_WIDTH = w  # 12
        self.MAZE_HEIGHT = h  # 12
        self.MAZE_DEPTH_TRAJECTORY = d  # 10
        self.LENGTH_E_CHAR = Ne_char
        self.NUM_RESIDUAL_BLOCKS = N_res_blocks

        self.INPUT_SHAPE = (self.MAX_TRAJECTORY_SIZE+1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY)
        self.TRAJECTORY_SHAPE = (self.MAX_TRAJECTORY_SIZE, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY) # 20x12x12x10
        self.CURRENT_STATE_SHAPE = (self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY-4)                      # 12x12x6

        # Create the model
        self.char_net = CharNet(input_tensor=self.TRAJECTORY_SHAPE,
                                n=self.NUM_RESIDUAL_BLOCKS,
                                N_echar=self.LENGTH_E_CHAR,
                                filters=filters)

        self.pred_net = PredNet(n=self.NUM_RESIDUAL_BLOCKS, filters=filters)

        # Set compilers / savers / loggers / callbacks

    def call(self, data):

        # To fix ERROR with Tensor <-> Numpy compatibility
        tf.compat.v1.enable_eager_execution()

        input_trajectory = data[0]   # input_traj            # inputs[..., 0:self.MAX_TRAJECTORY_SIZE, :, :, :]
        input_current_state = data[1] #  input_current    # inputs[..., self.MAX_TRAJECTORY_SIZE, :, :, 0:6]

        e_char = self.char_net(input_trajectory)

        # print("In ToMnet-N: ")
        # print("input_trajectory: ", input_trajectory.shape)
        # print("input_current_state: ", input_current_state.shape)
        # print("e_char SHAPE: ", e_char.shape)
        # print("e_char TYPE", type(e_char))
        # print("e_char", e_char)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 6) + (16, 8) ->
        # (16, 12, 12, 6) + (16, 8+4zero, 12repeat, 1) ->
        # (16, 12, 12, 7)   # NEW VERSION
        # (16, 12, 12, 8) + (16, 1, 12, 8) -> (16, 13, 12, 8)   # OLD VERSION
        # Spatialise and unite different data into one tensor
        # They are automatically decompose in the Pred Net to different data
        # --------------------------------------------------------------
        e_char_new = tf.concat(values=[e_char,e_char], axis = 1) # tf.repeat(e_char, repeats=2, axis=-1)
        e_char_new = e_char_new[..., 0:12]
        # print("Before Concatenation: ", e_char.shape)
        # print("After Concatenation: ", e_char_new.shape)

        # print("e_char_new: ", e_char_new.shape)
        e_char_new = tf.expand_dims(e_char_new, axis=-1)
        # print("e_char_new: ", e_char_new.shape)
        e_char_new = tf.repeat(e_char_new, repeats=12, axis=-1)
        # print("e_char_new: ", e_char_new.shape)
        e_char_new = tf.expand_dims(e_char_new, axis=-1)
        # print("e_char_new: ", e_char_new.shape)
        # print("input_current_state: ", input_current_state.shape)
        input_current_state = tf.cast(input_current_state, tf.float32)

        mix_data = tf.keras.layers.Concatenate(axis=-1)([input_current_state, e_char_new])

        # print("mix_data (pred input): ", mix_data.shape)

        pred = self.pred_net(mix_data)
        output = pred
        return output

    ### This is a trick to view shapes in summary() via
    ### model.model().summary()
    def model(self):
        x = tf.keras.Input(shape=self.INPUT_SHAPE)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))