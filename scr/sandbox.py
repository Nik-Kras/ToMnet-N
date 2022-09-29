from ToMnet_N import DataLoader
import os

loader = DataLoader.DataHandler(ts = 100,
                                w = 12,
                                h = 12,
                                d = 10)

directory = os.path.join('..', 'data', 'Saved Games', 'Experiment 1')
loader.load_all_games(directory=directory)
