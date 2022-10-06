#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class DataHandler(mp.ModelParameter):

The class for parsing txt data.

Note:
  Inherit mp.ModelParameter to share model constants.

@author: Chuang, Yun-Shiuan; Edwinn
"""

"""
The data stored like: 1x12x12x10. 1 - Time Step, 12x12 - Map Resolution, 10 - Depth (1 walls, 1 player, 4 goals, 4 actions)
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from random import shuffle
import re

class DataProcessor:

    def __init__(self, ts, w, h, d):
        self.MAX_TRAJECTORY_SIZE = ts # 20-50
        self.MAZE_WIDTH = w # 12
        self.MAZE_HEIGHT = h # 12
        self.MAZE_DEPTH = d # 10 (1player + 1wall + 4goals + 4 actions = 10)

        # Constants to keep track on standsrd
        # At which games are saved
        self.MAZE_LINE_START = 2
        self.MAZE_LINE_END = self.MAZE_WIDTH + 2
        self.CONSUMED_GOAL = self.MAZE_LINE_END + 1
        self.TRAJ_LENGTH = self.CONSUMED_GOAL + 1
        self.TRAJ_START = self.TRAJ_LENGTH + 1

    # It adds zeros at the beginning of the trajectories
    def zero_padding(self, max_elements, DictData):

        DataZeroPad = DictData.copy()

        for key, value in DictData.items():

            if key[-len("traj"):] == "traj":
                print("Apply Zero-Padding to " + key + "... ")
                N = len(DictData[key])
                # Make a uni-shape trajectories of max size 20x12x12x10
                DataZeroPad[key] = [np.zeros(shape=(max_elements,
                                                     self.MAZE_WIDTH,
                                                     self.MAZE_HEIGHT,
                                                     self.MAZE_DEPTH))] * N

                # Fill the last elements with real trajectory (implement pre-zero padding)
                for i in range(N):
                    Nt =  len(DictData[key][i]) # Number of real steps in the trajectory
                    if Nt > max_elements:
                        DataZeroPad[key][i] = DictData[key][i][-max_elements:]
                    else:
                        DataZeroPad[key][i][-Nt:,...] = DictData[key][i]

        print("Zero Padding was applied!")

        return DataZeroPad

    # Traj: 20x12x12x10
    # Cur: 12x12x6
    # ToMnet input: 21x12x12x10
    # Cur: 12x12x6 -> 1x12x12x10
    # Concatenate 20x12x12x10 + 1x12x12x10 -> 21x12x12x10
    def unite_traj_current(self, DictData):

        UniData = {
            "train_input": [np.zeros(shape=(self.MAX_TRAJECTORY_SIZE+1,
                                     self.MAZE_WIDTH,
                                     self.MAZE_HEIGHT,
                                     self.MAZE_DEPTH))] * len(DictData["train_traj"]),
            "test_input": [np.zeros(shape=(self.MAX_TRAJECTORY_SIZE + 1,
                                     self.MAZE_WIDTH,
                                     self.MAZE_HEIGHT,
                                     self.MAZE_DEPTH))] * len(DictData["test_traj"]),
            "valid_input": [np.zeros(shape=(self.MAX_TRAJECTORY_SIZE + 1,
                                    self.MAZE_WIDTH,
                                    self.MAZE_HEIGHT,
                                    self.MAZE_DEPTH))] * len(DictData["valid_traj"]),
            "train_goal": DictData["train_goal"],
            "test_goal": DictData["test_goal"],
            "valid_goal": DictData["valid_goal"],
            "train_act": DictData["train_act"],
            "test_act": DictData["test_act"],
            "valid_act": DictData["valid_act"]
        }

        print("-----")
        for key, value in UniData.items():

            if key[-len("input"):] == "input":

                print("Apply concatenation to " + key + "... ")
                # Add Trajectory in the beginning
                purpose = key[:-(len("input")+1)] # train / test / valid
                for i in range(len(DictData[purpose + "_traj"])):
                    UniData[key][i][0:self.MAX_TRAJECTORY_SIZE] = DictData[purpose + "_traj"][i]

                # Add Current in the end
                for i in range(len(DictData[purpose + "_traj"])):
                    # 12x12x6 -> 12x12x10
                    data_expanded = np.repeat(DictData[purpose + "_current"][i], repeats=2, axis=-1)
                    data_expanded = data_expanded[..., 0:10]
                    UniData[key][i][self.MAX_TRAJECTORY_SIZE] = data_expanded

        print("Concatenation is finished")
        return UniData

    def validate_data(self, DictData):

        """
        Data = {"train_traj":train_traj,
                "test_traj":test_traj,
                "valid_traj":valid_traj,
                "train_current":train_current,
                "test_current":train_current,
                "valid_current":valid_current,
                "train_goal": train_goal,
                "test_goal": test_goal,
                "valid_goal": valid_goal,
                "train_act": train_act,
                "test_act": test_act,
                "valid_act": valid_act}
        """

        print("----")
        print("Data validation... ")

        for key, value in DictData.items():

            if key[-len("traj"):] == "traj":
                if key == "train_traj":
                    self.trajectory_validation(value)
            elif key[-len("current"):] == "current":
                self.current_validation(value)
            elif key[-len("goal"):] == "goal":
                self.goal_validation(value)
            elif key[-len("act"):] == "act":
                self.act_validation(value)
            else:
                raise ValueError("Wrong key inside Data dictionary!")

        print("----")


    def trajectory_validation(self, traj):
        print("Trajectory validation... ")

        for index, tau in enumerate(traj):
            if index == 0:

                # Take first frame of trajectory
                frame_1 = tau[0]

                walls = frame_1[..., 0]
                player = frame_1[..., 1]
                goal1 = frame_1[..., 2]
                goal2 = frame_1[..., 3]
                goal3 = frame_1[..., 4]
                goal4 = frame_1[..., 5]

                fig, ax = plt.subplot_mosaic([
                    ["walls", "player"], ["goal 1", "goal 2"], ["goal 3", "goal 4"]
                ], constrained_layout=True, figsize=(16, 7))

                # Draw walls
                ax["walls"].set_title("Walls")
                ax["walls"].imshow(walls)

                # Draw Player
                ax["player"].set_title("Player")
                ax["player"].imshow(player)

                # Draw Goal 1
                ax["goal 1"].set_title("Goal 1")
                ax["goal 1"].imshow(goal1)

                # Draw Goal 2
                ax["goal 2"].set_title("Goal 2")
                ax["goal 2"].imshow(goal2)

                # Draw Goal 3
                ax["goal 3"].set_title("Goal 3")
                ax["goal 3"].imshow(goal3)

                # Draw Goal 4
                ax["goal 4"].set_title("Goal 4")
                ax["goal 4"].imshow(goal4)

                plt.show()



    def current_validation(self, traj):
        print("Current state validation... ")

    def goal_validation(self, traj):
        print("Goal validation... ")

    def act_validation(self, traj):
        print("Action validation... ")