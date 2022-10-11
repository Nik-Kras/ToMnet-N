
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
import matplotlib.pyplot as plt

from ToMnet_N import ToMnet
from ToMnet_N import DataLoader
from ToMnet_N import DataProcessing
from ToMnet_N import Layers as L

# --------------------------------------------------------
# CONSTANTS and Parameters
# --------------------------------------------------------
BATCH_SIZE = 16
ROW = 12
COL = 12
DEPTH = 10
MAX_TRAJ = 15
EPOCHS = 25 # 150 (no need to have more than 150)

MODEL_PATH = "../save_model/overfitted"

def dict_to_tensors(Dict):

    def make_y_outputs(folded_list):
        list_of_arrays = folded_list
        indices = list(np.concatenate([list_of_arrays], axis=0))
        indices = [x - 1 for x in indices]  # 1-4 --> 0-3
        depth = 4
        return tf.one_hot(indices, depth)

    X_Train = tf.convert_to_tensor(Dict["train_input"])
    X_Test = tf.convert_to_tensor(Dict["test_input"])
    X_Valid = tf.convert_to_tensor(Dict["valid_input"])

    Y_goal_Train = make_y_outputs(Dict["train_goal"])
    Y_goal_Test = make_y_outputs(Dict["test_goal"])
    Y_goal_Valid = make_y_outputs(Dict["valid_goal"])

    Y_act_Train = make_y_outputs(Dict["train_act"])
    Y_act_Test = make_y_outputs(Dict["test_act"])
    Y_act_Valid = make_y_outputs(Dict["valid_act"])

    return X_Train, X_Test, X_Valid, \
           Y_goal_Train, Y_goal_Test, Y_goal_Valid, \
           Y_act_Train, Y_act_Test, Y_act_Valid,

def save_game_to_draw(full_trajectory, predicted_actions):
    print("Puk-puk")

def load_data(directory):
    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    data_handler = DataLoader.DataHandler(ts=MAX_TRAJ,
                                          w=ROW,
                                          h=COL,
                                          d=DEPTH)
    #
    # single_game = {
    #     "traj": traj,  # Original trajectory
    #     "act": act,
    #     "goal": goal,
    #     "ToM":
    #         {
    #             "traj_history": traj_history,
    #             # Sequence of trajectories for ToMnet predictions. NO ZERO PADDING HERE!
    #             "current_state_history": current_state_history,
    #             "actions_history": actions_history
    #         }
    # }
    single_game = data_handler.load_one_game(directory=directory)

    # --------------------------------------------------------
    # 2. Pre-process data - Zero Padding
    # --------------------------------------------------------
    data_processor = DataProcessing.DataProcessor(ts=MAX_TRAJ,
                                                  w=ROW,
                                                  h=COL,
                                                  d=DEPTH)

    # data_processor.validate_data(Data)

    single_game = data_processor.zero_pad_single_game(max_elements=MAX_TRAJ,
                                                      single_game=single_game)

    single_game = data_processor.unite_single_traj_current(single_game)

    # Make Tensors from List
    indices = [x - 1 for x in single_game["ToM"]["actions_history"]]  # 1-4 --> 0-3
    depth = 4
    X_Train = tf.convert_to_tensor(single_game["united_input"])
    Y_act_Train = tf.one_hot(indices, depth)

    return X_Train, Y_act_Train

def train_model():

    TrainingAccuracy = pd.DataFrame()
    TrainHistory = pd.DataFrame()
    # --------------------------------------------------------
    # 3. Create and set the model
    # --------------------------------------------------------
    print("----")
    print("Create a model")
    Learning_Rate = 0.0001
    t = ToMnet.ToMnet(ts=MAX_TRAJ,
                      w=ROW,
                      h=COL,
                      d=DEPTH)
    t.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(Learning_Rate, clipnorm=1.0),
              # tf.keras.optimizers.Adam(learning_rate=0.0001)
              metrics=['accuracy'])

    t.fit(x=X_Train, y=Y_act_Train,
          epochs=1, batch_size=1, verbose=2)

    t.summary()

    # --------------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------------
    print("Train a Model")
    history = t.fit(x=X_Train, y=Y_act_Train,
                    epochs=EPOCHS, batch_size=1, verbose=2)

    TrainingAccuracy = TrainingAccuracy.append(pd.DataFrame({str(1): history.history["accuracy"]}))
    print("TrainingAccuracy", TrainingAccuracy)

    TrainHistory = TrainHistory.append(pd.DataFrame({
        "loss": history.history['loss'],
        "accuracy": history.history['accuracy']
    }))

    plt.plot(
        np.arange(1, EPOCHS + 1),
        history.history['loss'],
        label='Loss', lw=3
    )
    plt.plot(
        np.arange(1, EPOCHS + 1),
        history.history['accuracy'],
        label='Accuracy', lw=3
    )

    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    plt.show()

    TrainingAccuracy.to_csv('TrainingAccuracy.csv')
    TrainHistory.to_csv("TrainHistory_LR_Search.csv")

    t.save(MODEL_PATH)

def load_model():
    custom_layers = {
        "CustomCnn": L.CustomCnn,
        "ResBlock": L.ResBlock,
        "CustomLSTM": L.CustomLSTM,
    }
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_layers)

if __name__ == "__main__":

    # To fix ERROR: OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
    # OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # SHOULD DELETE?

    # Load data
    path_overfit = os.path.join('..', 'data', 'Saved Games', 'Overfit')
    X_Train, Y_act_Train = load_data(directory=path_overfit)

    # Train the model
    train_model()

    # Use trained model
    model =  load_model()

    history = model.fit(x=X_Train, y=Y_act_Train,
                    epochs=EPOCHS, batch_size=1, verbose=2)

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

