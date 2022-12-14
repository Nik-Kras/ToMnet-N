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

    def zero_pad_single_game(self, max_elements, single_game):

        # A single game has several trajectories
        all_trajectories = single_game["ToM"]["traj_history"]
        N = len(all_trajectories)
        TrajZeroPad = []

        for i in range(N):
            zero_pad_trajectory = np.zeros(shape=(max_elements,
                                        self.MAZE_WIDTH,
                                        self.MAZE_HEIGHT,
                                        self.MAZE_DEPTH))
            current_trajectory = all_trajectories[i]
            Nt = len(current_trajectory)  # Number of real steps in the current trajectory
            if Nt > max_elements:
                zero_pad_trajectory = current_trajectory[-max_elements:]
            else:
                zero_pad_trajectory[:Nt, ...] = current_trajectory
            TrajZeroPad.append(zero_pad_trajectory)

        single_game["ToM"]["traj_history_zp"] = TrajZeroPad

        return single_game


    # It adds zeros at the beginning of the trajectories
    def zero_padding(self, max_elements, DictData):

        DataZeroPad = DictData.copy()

        for key, value in DictData.items():

            if key[-len("traj"):] == "traj":
                print("Apply Zero-Padding to " + key + "... ")
                all_trajectories = DictData[key]
                N = len(all_trajectories)
                TrajZeroPad = []

                # Fill the last elements with real trajectory (implement pre-zero padding)
                for i in range(N):
                    zero_pad_trajectory = np.zeros(shape=(max_elements,
                                                          self.MAZE_WIDTH,
                                                          self.MAZE_HEIGHT,
                                                          self.MAZE_DEPTH))
                    current_trajectory = all_trajectories[i]
                    Nt =  len(current_trajectory) # Number of real steps in the trajectory
                    if Nt > max_elements:
                        zero_pad_trajectory = current_trajectory[-max_elements:]
                    else:
                        zero_pad_trajectory[:Nt, ...] = current_trajectory

                    # if key == "valid_traj":
                    #     actions = DictData["valid_act"]
                    #     ac = actions[i] # Getting an action TOMnet must predict
                    #     print("Next Action should be: ", ac)
                    #     self.one_trajectory_validation(zero_pad_trajectory)
                    #     cur_states = DictData["valid_current"]
                    #     cur_state = cur_states[i]
                    #     self.current_validation(cur_state)
                    TrajZeroPad.append(zero_pad_trajectory)

                DataZeroPad[key] = TrajZeroPad

        print("Zero Padding was applied!")

        return DataZeroPad

        # It adds zeros at the beginning of the trajectories
    def zero_padding_v2(self, max_elements, all_games):

        # all_games = {
        #     "traj_history": traj_history,
        #     "traj_history_zp": traj_history_zp                    # Trajectory with Zero Padding
        #     "current_state_history": current_state_history,
        #     "actions_history": actions_history
        # }

        uniform_shape = (1, max_elements, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH)

        zero_padded_trajectories = []   # ndarray, not list
        unfolded_current_states = []
        unfolded_action_history = []
        all_trajectories = all_games["traj_history"]
        all_current_states = all_games["current_state_history"]
        all_actions = all_games["actions_history"]
        N_all_games = len(all_trajectories)

        # Go one by one game
        # Where each game consist of many trajectories
        tracker_var = 0
        for i in range(N_all_games):

            traj = all_trajectories[i]
            cur = all_current_states[i]
            act = all_actions[i]
            N_traj = len(traj) # traj.shape[0]      # Number of trajectories in current game

            for j in range(N_traj):

                ### Init single piece of data from a game
                current_trajectory = traj[j]
                current_state = cur[j]
                current_action = act[j]

                ### Trajectory
                zero_pad_trajectory = np.zeros(shape=uniform_shape)
                Nt = current_trajectory.shape[0]  # Number of real steps in the trajectory

                # Save game in a bigger array so the rest is fiiled with zeros
                if Nt > max_elements:
                    zero_pad_trajectory[0, ...] = current_trajectory[-max_elements:]
                else:
                    zero_pad_trajectory[0, 0:Nt, ...] = current_trajectory

                zero_padded_trajectories.append(zero_pad_trajectory[0,...])

                ### Current state
                unfolded_current_states.append(current_state)

                ### Action
                unfolded_action_history.append(current_action)


            # Keep track on progress
            if i >= int(N_all_games * tracker_var / 100) - 2:
                print('Zero-Padded data ' + str(tracker_var) + '%')
                tracker_var += 5

        zero_padded_trajectories = np.array(zero_padded_trajectories)
        unfolded_current_states = np.array(unfolded_current_states)
        unfolded_action_history = np.array(unfolded_action_history)
        all_games["traj_history_zp"] = zero_padded_trajectories
        all_games["current_state_history"] = unfolded_current_states
        all_games["actions_history"] = unfolded_action_history

        print(all_games["traj_history_zp"].shape)

        print("Zero Padding was applied!")

        return all_games

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


    def unite_single_traj_current(self, single_game):

        print("Apply concatenation to a single trajectory... ")
        trajectories_list = single_game["ToM"]["traj_history"]
        current_state_list = single_game["ToM"]["current_state_history"]
        assert len(trajectories_list) == len(current_state_list)

        N = len(trajectories_list)
        united_data = []
        for i in range(N):

            cur_traj = trajectories_list[i]     # 15x12x12x10
            cur_state = current_state_list[i]   # 12x12x6

            cur_state_expanded = np.zeros(shape=(self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH))
            cur_state_expanded[..., 0:6] = cur_state # 12x12x6 -> 12x12x10

            Ntraj = cur_traj.shape[0]
            concat_shape = (Ntraj + 1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH)
            concatenated_data = np.zeros(shape=concat_shape)
            concatenated_data[0:Ntraj] = cur_traj
            concatenated_data[Ntraj] = cur_state_expanded

            united_data.append(concatenated_data)

        single_game["ToM"]["united_input"] = united_data

        return single_game


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

    def one_trajectory_validation(self, traj):

        for i in range(len(traj)):
            # Take i-th frame of trajectory
            frame_1 = traj[i]

            walls = frame_1[..., 0]
            player = frame_1[..., 1]
            goal1 = frame_1[..., 2]
            goal2 = frame_1[..., 3]
            goal3 = frame_1[..., 4]
            goal4 = frame_1[..., 5]
            act1 = frame_1[..., 6]
            act2 = frame_1[..., 7]
            act3 = frame_1[..., 8]
            act4 = frame_1[..., 9]

            to_draw = {
                "walls": walls,
                "player": player,
                "walls2": walls,
                "player2": player,
                "goal1": goal1,
                "goal2": goal2,
                "goal3": goal3,
                "goal4": goal4,
                "act1(UP)": act1,
                "act2(RIGHT)": act2,
                "act3(DOWN)": act3,
                "act4(LEFT)": act4
            }

            ROW = 3
            COL = 4
            fig, axs = plt.subplots(ROW, COL, figsize=(7, 6))
            row = 0
            col = 0
            for key, value in to_draw.items():
                axs[row, col].imshow(value)
                axs[row, col].set_title(key + "::" + str(i))
                axs[row, col].axis("off")
                col = col + 1
                if col == COL:
                    col = 0
                    row = row + 1
            plt.show()

    def trajectory_validation(self, traj):
        print("Trajectory validation... ")

        for index, tau in enumerate(traj):
            if index == 0:

                for i in range(tau.shape[0]):

                    # Take i-th frame of trajectory
                    frame_1 = tau[i]

                    walls = frame_1[..., 0]
                    player = frame_1[..., 1]
                    goal1 = frame_1[..., 2]
                    goal2 = frame_1[..., 3]
                    goal3 = frame_1[..., 4]
                    goal4 = frame_1[..., 5]
                    act1 = frame_1[..., 6]
                    act2 = frame_1[..., 7]
                    act3 = frame_1[..., 8]
                    act4 = frame_1[..., 9]

                    fig, ax = plt.subplot_mosaic([
                        ["walls",  "player"],
                        ["goal 1", "goal 2"],
                        ["goal 3", "goal 4"],
                        ["act 1",  "act 2"],
                        ["act 3", "act 4"]
                    ], constrained_layout=True)

                    # Draw walls
                    ax["walls"].set_title("Walls-" + str(i))
                    ax["walls"].imshow(walls)

                    # Draw Player
                    ax["player"].set_title("Player-" + str(i))
                    ax["player"].imshow(player)

                    # Draw Goal 1
                    ax["goal 1"].set_title("Goal 1-" + str(i))
                    ax["goal 1"].imshow(goal1)

                    # Draw Goal 2
                    ax["goal 2"].set_title("Goal 2-" + str(i))
                    ax["goal 2"].imshow(goal2)

                    # Draw Goal 3
                    ax["goal 3"].set_title("Goal 3-" + str(i))
                    ax["goal 3"].imshow(goal3)

                    # Draw Goal 4
                    ax["goal 4"].set_title("Goal 4-" + str(i))
                    ax["goal 4"].imshow(goal4)

                    # Draw Action 1
                    ax["act 1"].set_title("Action 1-" + str(i))
                    ax["act 1"].imshow(act1)

                    # Draw Action 2
                    ax["act 2"].set_title("Action 2-" + str(i))
                    ax["act 2"].imshow(act2)

                    # Draw Action 3
                    ax["act 3"].set_title("Action 3-" + str(i))
                    ax["act 3"].imshow(act3)

                    # Draw Action 4
                    ax["act 4"].set_title("Action 4-" + str(i))
                    ax["act 4"].imshow(act4)

                    plt.show()



    def current_validation(self, cur):
        print("Current state validation... ")

        to_draw = {
            "walls": cur[..., 0],
            "player": cur[..., 1],
            "walls2": cur[..., 0],
            "player2": cur[..., 1],
            "goal1": cur[..., 2],
            "goal2": cur[..., 3],
            "goal3": cur[..., 4],
            "goal4": cur[..., 5],
        }

        ROW = 2
        COL = 4
        fig, axs = plt.subplots(ROW, COL, figsize=(7, 6))
        row = 0
        col = 0
        for key, value in to_draw.items():
            axs[row, col].imshow(value)
            axs[row, col].set_title(key + ":: Current State")
            axs[row, col].axis("off")
            col = col + 1
            if col == COL:
                col = 0
                row = row + 1
        plt.show()

    def goal_validation(self, traj):
        print("Goal validation... ")

    def act_validation(self, traj):
        print("Action validation... ")