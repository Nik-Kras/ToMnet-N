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

import os
import sys
import numpy as np
from random import shuffle
import re

class DataHandler:

    def __init__(self, ts, w, h, d):
        self.MAX_TRAJECTORY_SIZE = ts # 20-50
        self.MAZE_WIDTH = w # 12
        self.MAZE_HEIGHT = h # 12
        self.MAZE_DEPTH_TRAJECTORY = d # 11

        # Constants to keep track on standsrd
        # At which games are saved
        self.MAZE_LINE_START = 2
        self.MAZE_LINE_END = self.MAZE_WIDTH + 2
        self.CONSUMED_GOAL = self.MAZE_LINE_END + 1
        self.TRAJ_LENGTH = self.CONSUMED_GOAL + 1
        self.TRAJ_START = self.TRAJ_LENGTH + 1

    # It loads full trajectory, sequence of actions and consumed goal per game
    def load_all_games(self, directory):

        # Get names of games
        files = os.listdir(directory)
        r = re.compile(".*.txt")
        files = list(filter(r.match, files))
        Nfiles = len(files)
        print("----")
        print("Saved Games found: ", Nfiles)
        print("Games names: ", files)

        # Save all trajectories and labels
        trajectories = [] # np.empty([1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY])
        actions = [] # np.empty(1)
        labels = [] # np.empty(1)

        # ------------------------------------------------------------------
        # Parse file one by one
        # ------------------------------------------------------------------
        j = 0  # for tracking progress (%)
        for i, file in enumerate(files):

            # Read one game
            traj, act, goal = self.read_one_game(filename=os.path.join(directory, file))

            # Append a game to data
            trajectories.append(traj)
            actions.append(act)
            labels.append(goal)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfiles / 100))-1:
                print('Parsed ' + str(j) + '%')
                j += 10
        print("----")

        # ------------------------------------------------------------------
        # Prepare data from games. -> Make many trajectories for each game
        # ------------------------------------------------------------------
        print("Augment data. One game creates many training samples!")

        data_trajectories = []
        data_current_state = []
        data_actions = []
        data_labels = []
        j = 0  # for tracking progress (%)

        # Process Game-per-Game
        for i in range(Nfiles):

            # Consider only games with more than 6 moves
            if trajectories[i].shape[0] < 6:
                continue

            # Prepare data from one game
            data_trajectories1, data_current_state1, data_actions1, data_labels1 = self.generate_data_from_game(trajectories=trajectories[i],                                                                                                 actions=actions[i],
                                                                                                                labels=labels[i])
            # Append to a single structure
            data_trajectories.append(data_trajectories1)
            data_current_state.append(data_current_state1)
            data_actions.append(data_actions1)
            data_labels.append(data_labels1)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfiles / 100))-1:
                print('Augmented data ' + str(j) + '%')
                j += 10

        print("----")

        # ------------------------------------------------------------------
        # Split the data  to Train / Test / Valid
        # ------------------------------------------------------------------
        print("Create training/testing/validation sets")

        train_traj, test_traj, valid_traj, \
        train_current, test_current, valid_current, \
        train_goal, test_goal, valid_goal, \
        train_act, test_act, valid_act = self.split_and_shaffle(data_trajectories=data_trajectories,
                                                                  data_current_state=data_current_state,
                                                                  data_actions=data_actions,
                                                                  data_labels=data_labels)

        print("----")

        # ------------------------------------------------------------------
        # Zero-Padding???
        # ------------------------------------------------------------------

        return train_traj, test_traj, valid_traj, \
               train_current, test_current, valid_current, \
               train_goal, test_goal, valid_goal, \
               train_act,  test_act,  valid_act

    # Returns trajectory, actions and consumed goal
    # For a single game
    def read_one_game(self, filename):
        '''
            Return
                traj - (ActionsInGame x MapWidth x MapHeight x MapDepth) (15x12x12x10)
                actions - (ActionsInGame) (array of numbers representing actions)
                goal - (ActionsInGame)  (array of the same goal *For Experiment 1*)
        '''

        traj = np.empty((1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY))
        act  = np.empty(1, dtype=np.int8)
        goal = np.empty(1, dtype=np.int8)

        # output.shape(100, 12, 12, 10) where 100 is Max Trajectory Size, 12x12 is WidthxHeight and 10 is Depth (1walls + 1player + 4goals + 4actions)
        output = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY, self.MAX_TRAJECTORY_SIZE))
        label = ''
        steps = []
        with open(filename) as fp:
            lines = list(fp)
            maze = lines[self.MAZE_LINE_START:self.MAZE_LINE_END]

            # Parse maze to 2d array, remove boundary walls.
            for i in range(self.MAZE_WIDTH):
                maze[i] = list(maze[i])
                maze[i] = maze[i][1:len(maze[i]) - 2]   # Transform: #row#\n -> row

            # Original maze (without walls)
            np_maze = np.array(maze)

            # Plane for obstacles
            np_obstacles = np.where(np_maze == '#', 1, 0).astype(np.int8)

            # Plane for agent's initial position
            np_agent = np.where(np_maze == 'O', 1, 0).astype(np.int8)

            # Plane for goals
            targets = ['A', 'B', 'C', 'D']  # for the simplified 4-targets mazes
            np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
            for target, i in zip(targets, range(len(targets))):
                np_targets[:, :, i] = np.where(np_maze == target, 1, 0)
            np_targets = np_targets.astype(int)

            # Save Consumed Goal
            goal_line = lines[self.CONSUMED_GOAL]
            _, goal_sym = goal_line.split(" : ")
            goal_sym = goal_sym[0]
            goal_num = self.goal_sym_to_num(goal_sym)

            # Get Trajectory Length
            Ntraj_line = lines[self.TRAJ_LENGTH]
            _, Ntraj = Ntraj_line.split(": ")
            Ntraj = int(Ntraj)

            # Save Actions & Save Trajectory

            trajectory = lines[self.TRAJ_START : self.TRAJ_START + Ntraj]
            agent_locations = []
            for i, tau in enumerate(trajectory):
                # Decompose
                tau = tau[:len(tau) - 1]  # Transform: 'output\n' -> 'output'
                tmp = tau.split(" : ")
                pos = tmp[0]
                pos = pos[1:-1]
                row, col = pos.split(", ")

                # Save
                # NOTE: first element in act & goal are trash values and MUST be replaced
                if i == 0:
                    agent_locations.append([int(row), int(col)])
                    act[0] = int(tmp[1])
                    goal[0] = goal_num  # self.sym_to_goal(tmp[2], consumed=)
                else:
                    agent_locations.append([int(row), int(col)])
                    act = np.append(act, int(tmp[1]))
                    goal = np.append(goal, goal_num) # self.sym_to_goal(tmp[2], consumed=)

                # Make Trajectory Tensor
                np_actions = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, 4), dtype=np.int8)
                a = act[i]
                np_actions[int(row), int(col), a] = 1

                np_tensor = np.dstack((np_obstacles, np_agent, np_targets, np_actions)) # (1walls + 1player + 4goals + 4actions)
                steps.append(np_tensor)
                traj = np.array(steps)

        fp.close()
        return traj, act, goal

    def goal_sym_to_num(self, goal_sym):
        out = 0
        if goal_sym == "A":
            out = 1
        elif goal_sym == "B":
            out = 2
        elif goal_sym == "C":
            out = 3
        elif goal_sym == "D":
            out = 4
        else:
            raise ValueError("ERROR: wrong goal sym was given!")
        return out

    # It deconstructs each game to a series of samples.
    # Single trajectory becomes a sequence of rising trajectories with same
    # Consumed goals
    def generate_data_from_game(self, trajectories, actions, labels):

        # Make full data from a game
        data_trajectories = []
        data_current_state = []
        data_actions = []
        data_labels = []

        MIN_ACTIONS = 5
        for i in range(MIN_ACTIONS, trajectories.shape[0]):
            data_trajectories.append(trajectories[0:i,...])     # Trajectory to the state
            data_current_state.append(trajectories[i,..., 0:6]) # Current state # (1walls + 1player + 4goals)
            data_actions.append(actions[i,...])                 # Next Action
            data_labels.append(labels[i,...])                   # Consumed Goal

        return data_trajectories, data_current_state, data_actions, data_labels

    def split_and_shaffle(self, data_trajectories, data_current_state, data_actions, data_labels):

        N_Total = len(data_trajectories)
        N_train = int(np.ceil(N_Total * 0.65))
        N_test  = int(np.ceil(N_Total * 0.20))
        N_valid = int(np.ceil(N_Total * 0.15))

        print("Total number of games after filtration: ", N_Total)
        print("Games for training: ", N_train)
        print("Games for testing: ", N_test)
        print("Games for validation: ", N_valid)

        total_indexes = list(range(N_Total))
        shuffle(total_indexes)

        train_indexes = total_indexes[0:N_train]
        test_indexes  = total_indexes[N_train:N_train+N_test]
        valid_indexes = total_indexes[N_train+N_test:]

        train_traj = [data_trajectories[i] for i in train_indexes]
        test_traj = [data_trajectories[i] for i in test_indexes]
        valid_traj = [data_trajectories[i] for i in valid_indexes]

        train_current = [data_current_state[i] for i in train_indexes]
        test_current = [data_current_state[i] for i in test_indexes]
        valid_current = [data_current_state[i] for i in valid_indexes]

        train_goal = [data_labels[i] for i in train_indexes]
        test_goal = [data_labels[i] for i in test_indexes]
        valid_goal = [data_labels[i] for i in valid_indexes]

        train_act = [data_actions[i] for i in train_indexes]
        test_act = [data_actions[i] for i in test_indexes]
        valid_act = [data_actions[i] for i in valid_indexes]

        # Unpack lists in data to be a single list of all games
        train_traj = sum(train_traj, [])
        test_traj = sum(test_traj, [])
        valid_traj = sum(valid_traj, [])

        train_current = sum(train_current, [])
        test_current = sum(test_current, [])
        valid_current = sum(valid_current, [])

        train_goal = sum(train_goal, [])
        test_goal = sum(test_goal, [])
        valid_goal = sum(valid_goal, [])

        train_act = sum(train_act, [])
        test_act = sum(test_act, [])
        valid_act = sum(valid_act, [])

        print("Time Steps for training: ", len(train_act))
        print("Time Steps for testing: ", len(test_act))
        print("Time Steps for validation: ", len(valid_act))

        return train_traj, test_traj, valid_traj, \
               train_current, test_current, valid_current, \
               train_goal, test_goal, valid_goal, \
               train_act, test_act, valid_act