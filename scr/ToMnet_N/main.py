
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

from TF2commented_batch_generator import BatchGenerator as bg
from DataLoader import DataHandler as dh
from ToMnet import ToMnet


if __name__ == "__main__":

    BATCH_SIZE = 16

    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    LIST_SUBJECTS = ["S033b"]
    path_txt_data = os.path.join('..', '..', 'data')
    dir = os.path.join(path_txt_data, LIST_SUBJECTS[0])

    data_handler = dh.DataHandler()

    train_data_traj, vali_data_traj, \
    test_data_traj, train_labels_traj, \
    vali_labels_traj, test_labels_traj, \
    all_files_traj, train_files_traj, \
    vali_files_traj, test_files_traj = \
        data_handler.parse_whole_data_set(dir,
                                          mode="all",
                                          shuf=True,
                                          subset_size=-1,
                                          parse_query_state=False)

    train_data_query_state, vali_data_query_state, \
    test_data_query_state, train_labels_query_state, \
    vali_labels_query_state, test_labels_query_state, \
    all_files_query_state, train_files_query_state, \
    vali_files_query_state, test_files_query_state = \
        data_handler.parse_whole_data_set(dir,
                                          mode="all",
                                          shuf=True,
                                          subset_size=-1,
                                          parse_query_state=True)

    batch_generator = bg.BatchGenerator()

    train_batch_data_traj, train_batch_labels_traj, \
    train_batch_data_query_state, train_batch_labels_query_state \
        = batch_generator.generate_train_batch(train_data_traj,
                                                    train_labels_traj,
                                                    train_data_query_state,
                                                    train_labels_query_state,
                                                    BATCH_SIZE)

    # --------------------------------------------------------------
    # Generate batches for validation data
    # --------------------------------------------------------------
    vali_batch_data_traj, vali_batch_labels_traj, \
    vali_batch_data_query_state, vali_batch_labels_query_state \
        = batch_generator.generate_vali_batch(vali_data_traj,
                                                   vali_labels_traj,
                                                   vali_data_query_state,
                                                   vali_labels_query_state,
                                                   BATCH_SIZE)

    # --------------------------------------------------------
    # 2. Pre-process data
    # --------------------------------------------------------

    # Split the data

    # Make a mixed data of trajectory and current state

    # --------------------------------------------------------
    # 3. Create and set the model
    # --------------------------------------------------------
    t = ToMnet()

    # --------------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 5. Evaluate the model
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 6. Save the model
    # --------------------------------------------------------


    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

