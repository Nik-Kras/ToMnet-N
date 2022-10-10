import os
import numpy as np
import unittest
import ToMnet_N.DataLoader as dl

### Real values for a given game
check_train_traj = None
check_test_traj = None
check_valid_traj = None
check_train_current = None
check_test_current = None
check_valid_current = None
check_train_goal = None
check_test_goal = None
check_valid_goal = None
check_train_act = None
check_test_act = None
check_valid_act = None

ts, w, h, d = TRAJ_SHAPE = [15, 12, 12, 10]
game_directory = os.path.join('..', 'data', 'Saved Games', 'UnitTest')

### Data Loader
class TestLoadOneGame(unittest.TestCase):
    def runTest(self):
        loader = dl.DataHandler(ts, w, h, d)
        train_traj, test_traj, valid_traj, \
        train_current, test_current, valid_current, \
        train_goal, test_goal, valid_goal, \
        train_act, test_act, valid_act = loader.load_all_games(game_directory)


        self.assertEqual(6, 6, "incorrect area")



def init_checking_values():

    MAZE = """
            #######-##-#
            ----D---##-#
            #-#####-##-#
            #-#####-##-#
            #-----B-##-#
            #-#####-##-#
            #A#####-##-#
            --C-----##--
            ####-##-##-#
            ####-##-##-#
            --------O--#
            #-########-#
            """
    print("Maze: ", MAZE)

    ### Read the Maze
    walls = np.zeros(shape=(12,12))
    player = np.zeros(shape=(12,12))
    goals = np.zeros(shape=(12, 12, 4))
    for row, line in enumerate(MAZE.splitlines()):
        print("Line: ", line)
        for col, sym in enumerate(line):
            print("Symbol: ", sym)
            if sym == '#':
                walls[row, col] = 1
            elif sym == '-':
                continue    # Ignore path as all layers initialised with zeros
            elif sym == 'O':
                player[row, col] = 1
            elif sym == 'A':
                goals[row, col, 0] = 1
            elif sym == 'B':
                goals[row, col, 1] = 1
            elif sym == 'C':
                goals[row, col, 2] = 1
            elif sym == 'D':
                goals[row, col, 3] = 1

    action_list = [3] + [0]*9 + [3]*3
    N = len(action_list)
    actions = np.zeros(shape=(N, 12, 12, 4))
    print("Action seq: ", action_list)

    current_position = np.where(player == 1)
    for i in range(N):
        currecnt_action = action_list[i]
        row, col = current_position

        actions[i]



if __name__ == "__main__":
    print("Unit Tests are running!")
    init_checking_values()
    unittest.main()