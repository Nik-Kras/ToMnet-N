import Environment
import Agent
import numpy as np
from MapGenerator.Grid import *

SIZE = 10
ROWS = 30
COLS = 30

game = Environment.GridWorld(tot_row = ROWS, tot_col = COLS)
game.render()

#Define the state matrix
Generator = Grid(SIZE)
state_matrix = Generator.GenerateMap() - 1
game.setStateMatrix(state_matrix)
game.setPosition()
game.render()

player = Agent.AgentRL(tot_row = ROWS, tot_col = COLS, actionsSize=4)
player.updateWorldObservation(game.getWorldState())
