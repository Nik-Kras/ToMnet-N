from ToMnet_N import DataLoader
import os

loader = DataLoader.DataHandler(ts = 100,
                                w = 12,
                                h = 12,
                                d = 10)

directory = os.path.join('..', 'data', 'Saved Games', 'Experiment 1')

train_traj, test_traj, valid_traj, \
train_current, test_current, valid_current, \
train_goal, test_goal, valid_goal, \
train_act, test_act, valid_act = loader.load_all_games(directory=directory)
