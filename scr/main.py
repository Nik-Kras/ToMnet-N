import Environment
import Agent
import numpy as np
from MapGenerator.Grid import *

from keras.layers import *
from keras.models import *

import tensorflow as tf

SIZE = 10
ROWS = 30
COLS = 30

game = Environment.GridWorld(tot_row = ROWS, tot_col = COLS)
# game.render()

#Define the state matrix
Generator = Grid(SIZE)
state_matrix = Generator.GenerateMap() - 1
game.setStateMatrix(state_matrix)
game.setPosition()
game.render()

game.step(0)
game.render()
game.step(1)
game.render()
game.step(2)
game.render()
game.step(3)
game.render()

# player = Agent.AgentRL(tot_row = ROWS, tot_col = COLS, actionsSize=4)
# player.updateWorldObservation(game.getWorldState())
#
# tensor_data = tf.convert_to_tensor(game.getWorldState())
# tensor_data = tf.expand_dims(tensor_data, axis=-1)
#
# print(player.choseAction())