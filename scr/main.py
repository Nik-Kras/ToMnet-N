import Environment
import numpy as np
from MapGenerator.Grid import *

SIZE = 10
ROWS = 30
COLS = 30

game = Environment.GridWorld(tot_row = ROWS, tot_col = COLS)
game.render()

#Define the state matrix
Generator = Grid(SIZE)
state_matrix = Generator.GenerateMap()
print("State Matrix:")
print(state_matrix)
game.setStateMatrix(state_matrix)
game.render()