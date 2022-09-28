import tensorflow as tf
from tensorflow import keras

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
    BATCH_SIZE = 16
    TRAJECTORY_SHAPE = (10, 12, 12, 11)
    CURRENT_STATE_SHAPE = (12, 12, 6)

    LENGTH_E_CHAR = 8
    NUM_RESIDUAL_BLOCKS = 5

    TRAIN_EMA_DECAY = 0.95
    INIT_LR = 0.0001

    def __init__(self):
        super(ToMnet, self).__init__(name="ToMnet-N")

        # Create the model
        self.char_net = CharNet(input_tensor=self.TRAJECTORY_SHAPE,
                                n=self.NUM_RESIDUAL_BLOCKS,
                                N_echar=self.LENGTH_E_CHAR)

        self.pred_net = PredNet(n=self.NUM_RESIDUAL_BLOCKS)

        # Set compilers / savers / loggers / callbacks

    def call(self, inputs):
        input_trajectory = inputs[..., 0:10, :, :, :]
        input_current_state = inputs[..., 10, :, :, 0:6]

        e_char = self.char_net(input_trajectory)

        print("In ToMnet-N: ")
        print("input_trajectory: ", input_trajectory.shape)
        print("input_current_state: ", input_current_state.shape)
        print("e_char: ", e_char.shape)

        # --------------------------------------------------------------
        # Paper codes
        # (16, 12, 12, 6) + (16, 8) ->
        # (16, 12, 12, 8) + (16, 1, 12, 8) -> (16, 13, 12, 8)
        # Spatialise and unite different data into one tensor
        # They are automatically decompose in the Pred Net to different data
        # --------------------------------------------------------------
        input_current_state = tf.repeat(input_current_state, repeats=2, axis=-1)
        input_current_state = input_current_state[..., 0:8]
        print("input_current_state: ", input_current_state.shape)

        e_char = tf.expand_dims(e_char, axis=1)
        print("e_char: ", e_char.shape)
        e_char = tf.expand_dims(e_char, axis=1)
        print("e_char: ", e_char.shape)
        e_char = tf.repeat(e_char, repeats=12, axis=2)
        print("e_char: ", e_char.shape)

        mix_data = tf.keras.layers.Concatenate(axis=1)([input_current_state, e_char])

        print("pred input: ", mix_data.shape)

        pred = self.pred_net(mix_data)
        output = pred
        return output

    ### This is a trick to view shapes in summary() via
    ### model.model().summary()
    def model(self):
        x = tf.keras.Input(shape=(11, 12, 12, 11))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))