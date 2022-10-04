
# -*- coding: utf-8 -*-
"""
class Model(mp.ModelParameter):

The class for training the ToMNET model.

Note:
  Inherit mp.ModelParameter to share model constants.

@author: Chuang, Yun-Shiuan; Edwinn
"""
import os
import sys
import time
import datetime
import pandas as pd
import tensorflow as tf
import argparse
import numpy as np

from ToMnet_N import ToMnet
from ToMnet_N import DataLoader
from ToMnet_N import DataProcessing

def unite_traj_current(trajectory, current):
    print("Done")

if __name__ == "__main__":

    # --------------------------------------------------------
    # CONSTANTS and Parameters
    # --------------------------------------------------------
    BATCH_SIZE = 16
    ROW = 12
    COL = 12
    DEPTH = 10
    MAX_TRAJ = 20

    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    data_handler = DataLoader.DataHandler(ts = MAX_TRAJ,
                                          w = ROW,
                                          h = COL,
                                          d = DEPTH)

    path_exper_1 = os.path.join('..', 'data', 'Saved Games', 'Experiment 1')

    # Load 4 types of data for 3 purposes
    # Purpose:
    #   - Training
    #   - Testing
    #   - Validating
    # Data:
    #   - Trajectory (from 5 to MAX_TRAJ-1 steps)
    #   - Current state (map at the end of Trajectory)
    #   - Goal eaten by the end of the game (for preference prediction)
    #   - Next action from Current position (for action prediction)
    train_traj, test_traj, valid_traj, \
    train_current, test_current, valid_current, \
    train_goal, test_goal, valid_goal, \
    train_act, test_act, valid_act = \
        data_handler.load_all_games(directory=path_exper_1)

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

    # --------------------------------------------------------
    # 2. Pre-process data - Zero Padding
    # --------------------------------------------------------
    data_processor = DataProcessing.DataProcessor(ts = MAX_TRAJ,
                                                  w = ROW,
                                                  h = COL,
                                                  d = DEPTH)

    Data = data_processor.zero_padding(max_elements= MAX_TRAJ,
                                              DictData=Data)

    DataProcessed = data_processor.unite_traj_current(Data)


    # Make Tensors from List
    print("Converting data to Tensors... ")
    X_Train = tf.convert_to_tensor(DataProcessed["train_input"])
    list_of_arrays = DataProcessed["train_goal"]
    indices = list(np.concatenate([list_of_arrays], axis=0))
    indices = [x - 1 for x in indices] # 1-4 --> 0-3
    depth = 4
    Y_Train = tf.one_hot(indices, depth)# tf.convert_to_tensor(DataProcessed["train_goal"])
    print("Converting is finished")

    # --------------------------------------------------------
    # 3. Create and set the model
    # --------------------------------------------------------
    print("----")
    print("Create a model")
    t = ToMnet.ToMnet(ts = MAX_TRAJ,
                      w = ROW,
                      h = COL,
                      d = DEPTH)
    t.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # --------------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------------
    print("Train a Model")
    t.fit(x=X_Train, y=Y_Train,
          epochs=15, batch_size=10, verbose=2)

    # --------------------------------------------------------
    # 5. Evaluate the model
    # --------------------------------------------------------
    _, accuracy = t.evaluate(x=X_Train, y=Y_Train)
    print('Accuracy: %.2f' % (accuracy * 100))

    # --------------------------------------------------------
    # 6. Save the model
    # --------------------------------------------------------


    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

