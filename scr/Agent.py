import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *

class AgentRL:

    def __init__(self, tot_row=30, tot_col=30, actionsSize=4):
        self.action_space_size = actionsSize
        self.world_row = tot_row
        self.world_col = tot_col
        #The world is a matrix of size row x col x 6 (player, wall, goal 1, goal 2, goal 3, goal 4)
        #However, it could be represented as matrix row x col x 1, where
        # -1 for states which are not walkable
        # 0 for all the walkable states (non terminal)
        # [+1, +2, +3, +4] - for 4 different goals
        # NOTE: in future development the observability could be limited to, for example, 7x7 area
        self.observedWorld = np.zeros(shape=(tot_row, tot_col))
        # self.observedWorld = np.zeros(shape=(tot_row, tot_col, 6))  # Hot-end version

        # Create a model object
        self.model = ModelRL(actionsSize, tot_row, tot_col)

    def choseAction(self):
        tensor_data = tf.convert_to_tensor(self.observedWorld)
        tensor_data = tf.expand_dims(tensor_data, axis = -1)    # add one dimention for features
        tensor_data = tf.expand_dims(tensor_data, axis =  0)    # add one dimention for batch size

        # tensor_data = tf.reshape(tensor_data, shape=(1,30,30))

        print(tensor_data)
        print(tensor_data.shape)

        q_values = self.model.call(tensor_data)[0]
        action = tf.argmax(q_values)
        print("Q-values: ", q_values)
        print("The best action: ", action)
        return action

    def updateWorldObservation(self, newWorld):
        if np.shape(newWorld) == np.shape(self.observedWorld):
            self.observedWorld = newWorld
            plt.title("World Map")
            plt.axis('off')
            plt.imshow(self.observedWorld)
            plt.show()
        else:
            print("Wrong shape of the new world state!")

class ModelRL:

    def __init__(self, actionsSize, tot_row=30, tot_col=30):
        super().__init__()

        self.model = tf.keras.Sequential([
            Input(shape=(tot_row, tot_col, 1)),                     # batch_size (None), width, high, features!
            Conv2D(filters=16, kernel_size=(3, 3), padding="same"),
            ReLU(),
            Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            ReLU(),
            MaxPooling2D(),
            Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            ReLU(),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            ReLU(),
            MaxPooling2D(),

            Flatten(),

            Dense(128),
            ReLU(),
            Dense(32),
            ReLU(),
            Dense(units=actionsSize),
            Softmax()
        ])

    """
        Use the model architecture to get the output
    """
    def call(self, inputs):

        return self.model.predict(inputs)