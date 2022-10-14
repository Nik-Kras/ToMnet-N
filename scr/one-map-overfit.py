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
EPOCHS = 50 # 150 (no need to have more than 150)

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
    #             "traj_history_zp": traj_history_zp # Trajectory with Zero Padding
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

    # single_game = data_processor.unite_single_traj_current(single_game)

    single_game = data_processor.zero_pad_single_game(max_elements=MAX_TRAJ-1,
                                                      single_game=single_game)

    # Make Tensors from List
    indices = [x - 1 for x in single_game["ToM"]["actions_history"]]  # 1-4 --> 0-3
    depth = 4
    X_train_traj = tf.convert_to_tensor(single_game["ToM"]["traj_history_zp"])
    X_train_current = tf.convert_to_tensor(single_game["ToM"]["current_state_history"])
    Y_act_Train = tf.one_hot(indices, depth)

    # return X_Train, Y_act_Train
    return X_train_traj, X_train_current, Y_act_Train

def train_model(X_train_traj, X_train_current, Y_act_Train):

    # --------------------------------------------------------
    # 3. Create and set the model
    # --------------------------------------------------------
    print("----")
    print("Create a model")
    Learning_Rate = 0.0001
    t = ToMnet.ToMnet(ts=MAX_TRAJ-1,    # 14 frames are real trajectory, 1 frame is current state. So MAX_TRAJ is 14, not 15
                      w=ROW,
                      h=COL,
                      d=DEPTH)
    t.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(Learning_Rate, clipnorm=1.0),
              metrics=['accuracy'])

    t.fit(x=[X_train_traj, X_train_current], y=Y_act_Train,
          epochs=1, batch_size=1, verbose=2)

    t.summary()

    # --------------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------------
    print("Train a Model")
    history = t.fit(x=[X_train_traj, X_train_current], y=Y_act_Train,
                    epochs=EPOCHS, batch_size=1, verbose=2)
    plot_history(history)
    save_history(history)

    t.save(MODEL_PATH)

def plot_history(history):

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
    # 1. Build Initial Map (simple map) for rendering
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
    map_df.to_csv(path_to_save + str("simple_map.csv"))

    # --------------------------------------------------------
    # 2. Save Full Trajectory
    # --------------------------------------------------------
    full_traj_actions = []

    # Create list of actions saved in trajectory
    TS = input_data.shape[1] - 1   # Trajectory Size. The last frame is current state, no actions are shown there
    for i in range(TS):
        all_action_layers = np.array(input_data[0, i, ..., 6:10])

        # Find which action was performed
        bool_val = False
        for action_number in range(4):
            action_layer = np.array(all_action_layers[..., action_number], dtype=np.int8)
            max_val = action_layer.max()
            bool_val = 1 in action_layer    # np.where(n_array == 1) # Should also work
            bool_val = np.any(bool_val)
            if bool_val:
                full_traj_actions.append(action_number)
                break

        # If no actions were found in a frame - it is a Zero_padding. Finish here
        if not bool_val:
            TS = i
            break

    print(full_traj_actions)

    # Find initial position
    initial_coordinates = list(np.where(player_layer == 1))

    # Create coordinate sequence
    full_traj_coordinates = [initial_coordinates]
    for i in range(TS):
        coordinates = full_traj_coordinates[-1].copy()
        applied_action = full_traj_actions[i]

        dr = 0
        dc = 0
        if applied_action == 0:
            dr = -1
            dc = 0
        elif applied_action == 1:
            dr = 0
            dc = 1
        elif applied_action == 2:
            dr = 1
            dc = 0
        elif applied_action == 3:
            dr = 0
            dc = -1

        coordinates[0] = coordinates[0] + dr
        coordinates[1] = coordinates[1] + dc

        full_traj_coordinates.append(coordinates)

    print(full_traj_coordinates)
    full_traj_coordinates_df = pd.DataFrame(full_traj_coordinates)
    full_traj_coordinates_df.to_csv(path_to_save + str("full_traj.csv"))

    # --------------------------------------------------------
    # 3. Save Initial Trajectory
    # --------------------------------------------------------

    Nfull = len(full_traj_coordinates)
    if Nfull - predict_steps < 5:
        raise ValueError("The game is too short! It has only " + str(Nfull) + " moves, while you ask to predict"
                         + str(predict_steps) +
                         " actions. Give at least a game with trajectory length bigger than predicted actions by 5.")
    initial_traj_coordinates = full_traj_coordinates[0:-predict_steps]
    initial_traj_coordinates_df = pd.DataFrame(initial_traj_coordinates)
    initial_traj_coordinates_df.to_csv(path_to_save + str("init_traj.csv"))

    # --------------------------------------------------------
    # 4. Save Predicted Trajectory
    # --------------------------------------------------------

    # Initial trajectory for ToMnet
    zero_pad_shape = (predict_steps, ROW, COL, DEPTH)
    zaro_pad = tf.cast(tf.zeros(shape=zero_pad_shape), tf.float32)                  # For concat        #
    input_to_concat = tf.cast(input_data[0, 0:-predict_steps, ...], tf.float32)     # Data must be the same dtype
    initial_input_data = tf.concat(values=[input_to_concat, zaro_pad], axis=0)
    initial_input_data = tf.expand_dims(initial_input_data, axis=0)
    print("input_data shape: ", input_data.shape)
    print("initial_input_data shape: ", initial_input_data.shape)

    # Make action predictions
    predicted_actions = []
    current_player_coordinates = initial_traj_coordinates[-1].copy()
    coordinates = [current_player_coordinates]
    input_data = initial_input_data
    for i in range(predict_steps):
        # Get predicted action
        predict_distribution = model.predict(input_data)
        predicted_action = np.where(predict_distribution == predict_distribution.max())[1][0]   # Output: 0 - 3
        predicted_actions.append(predicted_action)

        # Update players coordinates
        player_position = current_player_coordinates.copy()
        if predicted_action == 0:    player_position[0] = player_position[0] - 1
        elif predicted_action == 1:  player_position[1] = player_position[1] + 1
        elif predicted_action == 2:  player_position[0] = player_position[0] + 1
        elif predicted_action == 3:  player_position[1] = player_position[1] - 1
        current_player_coordinates = player_position.copy()
        coordinates.append(current_player_coordinates)

        new_player_map = np.zeros(shape=(ROW, COL, 1))
        new_player_map[player_position[0], player_position[1], 0] = 1
        new_player_map = tf.convert_to_tensor(new_player_map, dtype=tf.float32)

        # Update action layers (ACTION IS ASSIGNED TO CURRENT FRAME, NOT NEW FRAME!!!)
        new_action_map = tf.zeros(shape=(ROW, COL, 4), dtype=tf.float32)
        old_action_map = np.zeros(shape=(ROW, COL, 4))
        old_action_map[player_position[0], player_position[1], predicted_action] = 1
        old_action_map = tf.convert_to_tensor(old_action_map, dtype=tf.float32)

        # Get wall layer
        wall_map = initial_input_data[0, 0, ..., 0]
        wall_map = tf.expand_dims(wall_map, axis=-1)

        # Get goals map
        goal_map = initial_input_data[0, 0, ..., 2:6]

        # Update current frame (Add assigned action to it!)
        steps_unfound = -predict_steps + i  # -5 -> none were found. -1 -> one is left to be found
        updated_traj = input_data[0, 0:steps_unfound, ...].numpy()
        updated_traj[-1, ..., 6:10] = old_action_map
        updated_traj = tf.convert_to_tensor(updated_traj, dtype=tf.float32)

        # Create new frame
        new_frame = tf.concat(values=[wall_map, new_player_map, goal_map, new_action_map], axis=-1)
        new_frame = tf.expand_dims(new_frame, axis=0)

        # Update input data trajectory with a new frame instead of zero-pad layer
        new_traj = tf.concat(values=[updated_traj, new_frame], axis=0)
        NeededZeros = MAX_TRAJ - new_traj.shape[0] - 1 # Add Zeros up to MAX_TRAJ-1, as last frame - current state!

        if NeededZeros > 0:
            zero_pad_shape = (NeededZeros, ROW, COL, DEPTH)
            zero_pad = tf.zeros(shape=zero_pad_shape, dtype=tf.float32)
            input_data = tf.concat(values=[new_traj, zero_pad], axis=0)
        else:
            input_data = new_traj
        input_data = tf.expand_dims(input_data, axis=0) # Add Batch Shape dim

        input_data.numpy() # For debug

        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

    coordinates_df = pd.DataFrame(coordinates)
    coordinates_df.to_csv(path_to_save + str("predicted_traj.csv"))

    return predicted_actions

if __name__ == "__main__":

    # To fix ERROR
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ### Load data
    X_train_traj, X_train_current, Y_act_Train = load_data(directory=GAME_PATH)

    ### Train the model
    train_model(X_train_traj, X_train_current, Y_act_Train)

    ### Use trained model
    # model =  load_model()

    ### Keep training the model
    # history = model.fit(x=X_Train, y=Y_act_Train,
    #                 epochs=EPOCHS, batch_size=1, verbose=2)
    # plot_history(history)

    ### Test it on one prediction

    """
    # Find an index for a random game longer than 10 steps
    N_trajectories = X_Train.shape[0]  # Number of whole trajectories saved from all games
    N = 0
    random_inx = 0
    while N < 10:
        random_inx = np.random.randint(0, N_trajectories-1)
        input_data = X_Train[random_inx, ...]

        # Measure real length of the traj (without Zero-Padding)
        for i in range(input_data.shape[0]):
            N = i
            # Check the presence of player's position on the map
            player_layer = np.array(input_data[i, ..., 1], dtype=np.int8)
            bool_val = 1 in player_layer  # np.where(n_array == 1) # Should also work
            bool_val = np.any(bool_val)
            if bool_val:
                pass
    """

    # Pick the longest trajectory, which has 14 moves and 15tf frame is current state
    # input_data = X_Train[-1, ...]
    # input_data = tf.expand_dims(input_data, axis=0)  # Add axis for "batch_size"
    # actual_action = Y_act_Train[-1]
    # yhat = model.predict(input_data)
    #
    # print("Actual action: ", actual_action)
    # print("Predicted action: ", yhat)

    ### Predict trajectory
    # predict_game(model, input_data, predict_steps=5)

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

