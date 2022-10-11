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
GAME_PATH = os.path.join('..', 'data', 'Saved Games', 'Overfit')

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
    plot_history(history)
    save_history(history)

    t.save(MODEL_PATH)

def plot_history(history):
    TrainHistory = pd.DataFrame()
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

def save_history(history, name="TrainHistory.csv"):
    TrainHistory = pd.DataFrame()
    TrainHistory = TrainHistory.append(pd.DataFrame({
        "loss": history.history['loss'],
        "accuracy": history.history['accuracy']
    }))
    TrainHistory.to_csv(name)


def load_model():
    custom_layers = {
        "CustomCnn": L.CustomCnn,
        "ResBlock": L.ResBlock,
        "CustomLSTM": L.CustomLSTM,
    }
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_layers)

def predict_game(model, input_data, predict_steps=5):

    # Trajectory Depth saved as - (np_obstacles, np_agent, np_targets, np_actions)
    # Traj shape = BS x TS x W x H x D. 1x5-15x12x12x10

    # Check for Batch_Size dim
    if input_data.shape[0] != 1:
        input_data = tf.expand_dims(input_data, axis=0)

    path_to_save = "../Results/Predictions/Prediction 1/"
    width = input_data.shape[2]
    height = input_data.shape[3]

    # --------------------------------------------------------
    # 1. Build Initial Map
    # --------------------------------------------------------
    simple_map = np.zeros((12, 12), dtype=np.int16)  # 0-path, 1-wall, 2/5-goals, 10-player

    # Put walls
    walls_layer = input_data[0, 0, ..., 0]
    for row in range(width):
        for col in range(height):
            if walls_layer[row, col] == 1:
                simple_map[row, col] = 1

    # Put player
    player_layer = input_data[0, 0, ..., 1]
    for row in range(width):
        for col in range(height):
            if player_layer[row, col] == 1:
                simple_map[row, col] = 10

    # Put goals
    goal_layer = input_data[0, 0, ..., 2:6]
    assert goal_layer.shape[-1] == 4            # Check that there are 4 layers for 4 goals
    for row in range(width):
        for col in range(height):
            if goal_layer[row, col, 0] == 1:
                simple_map[row, col] = 2
            elif goal_layer[row, col, 1] == 1:
                simple_map[row, col] = 3
            elif goal_layer[row, col, 2] == 1:
                simple_map[row, col] = 4
            elif goal_layer[row, col, 3] == 1:
                simple_map[row, col] = 5
    map_df = pd.DataFrame(simple_map)
    map_df.to_csv(path_to_save+str("simple_map.csv"))

    # --------------------------------------------------------
    # 2. Save Initial Trajectory
    # --------------------------------------------------------
    init_traj_actions = []

    # Create list of actions saved in trajectory
    TS = input_data.shape[1]    # Trajectory Size
    for i in range(TS):
        all_action_layers = input_data[0, 0, ..., 6:10]
        for action_number, action_layer in enumerate(all_action_layers):
            action_performed = 1 in action_layer
            if action_performed:
                init_traj_actions.append(action_number)
    print(init_traj_actions)

    # Find initial position
    initial_coordinates = np.where(player_layer == 1)

    # Create coordinate sequence
    init_traj_coordinates = [initial_coordinates]
    for i in range(TS):
        init_traj_coordinates

    # --------------------------------------------------------
    # 3. Save Predicted Trajectory
    # --------------------------------------------------------
    # Make action predictions
    predicted_actions = []
    coordinates = []
    for i in range(predict_steps):
        predict_distribution = model.predict(input_data)
        predicted_action = list(np.where(predict_distribution == max(predict_distribution)))    # 0 - 3
        predicted_actions.append(predicted_action)

        player_position = input_data

    return predicted_actions

if __name__ == "__main__":

    # To fix ERROR
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ### Load data
    X_Train, Y_act_Train = load_data(directory=GAME_PATH)

    ### Train the model
    # train_model()

    ### Use trained model
    model =  load_model()

    ### Keep training the model
    # history = model.fit(x=X_Train, y=Y_act_Train,
    #                 epochs=EPOCHS, batch_size=1, verbose=2)
    # plot_history(history)

    ### Test it on one prediction
    N_trajectories = X_Train.shape[0]
    random_inx = np.random.randint(0, N_trajectories-1)
    input_data = X_Train[random_inx, ...]
    input_data = tf.expand_dims(input_data, axis=0) # Add axis for "batch_size"
    actual_action = Y_act_Train[random_inx]
    yhat = model.predict(input_data)

    print("Actual action: ", actual_action)
    print("Predicted action: ", yhat)


    ### Predict trajectory
    predict_game(model, input_data, predict_steps=5)

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

