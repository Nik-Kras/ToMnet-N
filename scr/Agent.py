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

        q_values = self.model.call(tensor_data)
        action = max(q_values)
        return q_values

    def updateWorldObservation(self, newWorld):
        if np.shape(newWorld) == np.shape(self.observedWorld):
            self.observedWorld = newWorld
            #plt.title("World Map")
            #plt.axis('off')
            #plt.imshow(self.observedWorld)
            #plt.show()
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
        self.inputLayer = Input(shape=(tot_row, tot_col))
        self.conv1 = Conv2D(filters=16, kernel_size=(3, 3), padding="same")
        self.reluConv1 = ReLU()
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.reluConv2 = ReLU()
        self.pool1 = MaxPooling2D()
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.reluConv3 = ReLU()
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.reluConv4 = ReLU()
        self.pool2 = MaxPooling2D()

        self.flatten = Flatten()

        self.dense1 = Dense(128)
        self.reluDense1 = ReLU()
        self.dense2 = Dense(32)
        self.reluDense2 = ReLU()
        self.dense3 = Dense(actionsSize)
        self.outputLayer = Softmax()
        """

    """
        Use the model architecture to get the output
    """
    def call(self, inputs):

        return self.model.predict(inputs)
        """
        x = self.inputLayer(inputs)
        x = self.conv1(x)
        x = self.reluConv1(x)
        x = self.conv2(x)
        x = self.reluConv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.reluConv3(x)
        x = self.conv4(x)
        x = self.reluConv4(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.reluDense1(x)
        x = self.dense2(x)
        x = self.reluDense2(x)
        x = self.dense3(x)
        x = self.outputLayer(x)
        return x

        """