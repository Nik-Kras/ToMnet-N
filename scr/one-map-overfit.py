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
N_ECHAR = 8
N_RESBLOCKS = 32
LEARNING_RATE = 0.0001 / 5
BATCH_SIZE = 32
ROW = 12
COL = 12
DEPTH = 10
MAX_TRAJ = 15
EPOCHS = 100 # 150 (no need to have more than 150)

LOAD_PERCENTAGE = 0.05 # 0.1% = 5 games. 0.02% = 1 game


MODEL_PATH = "../save_model/overfitted"
TESTING_GAME_PATH = os.path.join('..', 'data', 'Saved Games', 'Overfit')
TRAINING_GAMES_PATH = os.path.join('..', 'data', 'Saved Games', 'Experiment 2')

def save_game_to_draw(full_trajectory, predicted_actions):
    print("Puk-puk")

def load_training_games(directory, load_percentage=0.2):
    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    data_handler = DataLoader.DataHandler(ts=MAX_TRAJ,
                                          w=ROW,
                                          h=COL,
                                          d=DEPTH)
    #
    # all_games = {
    #     "traj_history": traj_history,
    #     "traj_history_zp": traj_history_zp                    # Trajectory with Zero Padding
    #     "current_state_history": current_state_history,
    #     "actions_history": actions_history
    # }
    all_games = data_handler.load_all_games_v2(directory=directory, use_percentage=load_percentage)

    # --------------------------------------------------------
    # 2. Pre-process data - Zero Padding
    # --------------------------------------------------------
    data_processor = DataProcessing.DataProcessor(ts=MAX_TRAJ,
                                                  w=ROW,
                                                  h=COL,
                                                  d=DEPTH)

    all_games = data_processor.zero_padding_v2(max_elements=MAX_TRAJ,
                                                all_games=all_games)

    # Make Tensors from List
    indices = all_games["actions_history"]
    depth = 4
    X_train_traj = tf.convert_to_tensor(all_games["traj_history_zp"], dtype=tf.float32)
    X_train_current = tf.convert_to_tensor(all_games["current_state_history"], dtype=tf.float32)
    Y_act_Train = tf.one_hot(indices, depth)

    # return X_Train, Y_act_Train
    return X_train_traj, X_train_current, Y_act_Train

def load_one_game(directory):
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

    single_game = data_processor.zero_pad_single_game(max_elements=MAX_TRAJ,
                                                      single_game=single_game)

    # Make Tensors from List
    indices = [x - 1 for x in single_game["ToM"]["actions_history"]]  # 1-4 --> 0-3
    depth = 4
    X_train_traj = tf.convert_to_tensor(single_game["ToM"]["traj_history_zp"], dtype=tf.float32)
    X_train_current = tf.convert_to_tensor(single_game["ToM"]["current_state_history"], dtype=tf.float32)
    Y_act_Train = tf.one_hot(indices, depth)

    # return X_Train, Y_act_Train
    return X_train_traj, X_train_current, Y_act_Train

def train_model(X_train_traj, X_train_current, Y_act_Train):

    # --------------------------------------------------------
    # 3. Create and set the model
    # --------------------------------------------------------
    print("----")
    print("Create a model")

    t = ToMnet.ToMnet(ts=MAX_TRAJ,
                      w=ROW,
                      h=COL,
                      d=DEPTH,
                      Ne_char=N_ECHAR,
                      N_res_blocks=N_RESBLOCKS,
                      filters=64)
    t.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0),
              metrics=['accuracy'])

    t.fit(x=[X_train_traj, X_train_current], y=Y_act_Train,
          epochs=1, batch_size=BATCH_SIZE, verbose=2)

    t.summary()

    # --------------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------------
    print("Train a Model")
    history = t.fit(x=[X_train_traj, X_train_current], y=Y_act_Train,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    plot_history(history)
    save_history(history)

    # t.save(MODEL_PATH)

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

# I am only sending a trajectory here
# This trajectory will be divided to Traj and Current state
# And therefore coordinates will be calculated
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

    print("Full trajectory in actions: ", full_traj_actions)

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

    print("Full trajectory in coordinates: ", full_traj_coordinates)
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

    # Remove Zero-Padding from Trajectory
    # ...
    # Currently it doesn't have zero-padding

    # Traj -> Traj_zero_pad + current state
    input_traj = tf.cast(input_data[0, 0:-predict_steps, ...], tf.float32)
    input_current = tf.cast(input_data[0, -predict_steps, ..., 0:6], tf.float32)
    NeededZeros = MAX_TRAJ - input_traj.shape[0] # Add Zeros up to MAX_TRAJ
    if NeededZeros > 0:
        zero_pad_shape = (NeededZeros, ROW, COL, DEPTH)
        zero_pad = tf.zeros(shape=zero_pad_shape, dtype=tf.float32)
        input_traj = tf.concat(values=[input_traj, zero_pad], axis=0)
    input_traj = tf.expand_dims(input_traj, axis=0)
    input_current = tf.expand_dims(input_current, axis=0)

    input_traj    = tf.cast(input_traj, tf.float32)
    input_current = tf.cast(input_current, tf.float32)

    # Get initial coordinates
    np_input_current = input_current.numpy()
    np_player_postition = np_input_current[0, ..., 1]
    position = list(np.where(np_player_postition == np_player_postition.max()))

    # Make action predictions
    predicted_actions = []
    current_player_coordinates = initial_traj_coordinates[-1].copy()
    coordinates = [position] # [current_player_coordinates]
    for i in range(predict_steps):
        # Get predicted action
        predict_distribution = model.predict([input_traj, input_current])
        predicted_action = np.where(predict_distribution == predict_distribution.max())[1][0]   # Output: 0 - 3
        predicted_actions.append(predicted_action)
        print("Predicted Action: ", predicted_action)

        np_input_traj = input_traj.numpy()
        np_input_current = input_current.numpy()

        # --------------------------------------------------------
        # 4.1 Update layers
        # --------------------------------------------------------

        # Update players coordinates
        old_player_position = coordinates[-1].copy()    # current_player_coordinates.copy()
        player_position = old_player_position.copy()
        if predicted_action == 0:    player_position[0] = player_position[0] - 1
        elif predicted_action == 1:  player_position[1] = player_position[1] + 1
        elif predicted_action == 2:  player_position[0] = player_position[0] + 1
        elif predicted_action == 3:  player_position[1] = player_position[1] - 1

        # Check for safety (map boundaries)
        if player_position[0] > ROW-1: player_position[0] = ROW-1
        if player_position[0] < 0: player_position[0] = 0
        if player_position[1] > COL-1: player_position[1] = COL-1
        if player_position[1] < 0: player_position[1] = 0

        current_player_coordinates = player_position.copy()
        coordinates.append(current_player_coordinates)

        new_player_map = np.zeros(shape=(ROW, COL, 1))
        new_player_map[player_position[0], player_position[1], 0] = 1
        # new_player_map = tf.convert_to_tensor(new_player_map, dtype=tf.float32)

        # Update action layers (ACTION IS ASSIGNED TO CURRENT FRAME, NOT NEW FRAME!!! So it takes old player's position)
        # The old position is saved in Current State
        action_map = np.zeros(shape=(ROW, COL, 4))
        action_map[old_player_position[0], old_player_position[1], predicted_action] = 1
        # old_action_map = tf.convert_to_tensor(old_action_map, dtype=tf.float32)

        # Get wall layer
        wall_map = input_traj[0, 0, ..., 0]
        wall_map = tf.expand_dims(wall_map, axis=-1)
        np_wall_map = wall_map.numpy()

        # Get goals map
        goal_map = input_traj[0, 0, ..., 2:6]
        np_goal_map = goal_map.numpy()

        # --------------------------------------------------------
        # 4.2 Update Trajectory
        # --------------------------------------------------------
        # Indexes:  traj1, traj2, traj3, traj4, traj5, zp,    zp, zp, zp, zp
        # Become:   traj1, traj2, traj3, traj4, traj5, traj6, zp, zp, zp, zp
        # Initial number of zp = predict_steps
        # It decreases with increasing of i
        # Therefore I must update input_traj[0, -predict_step+i, ...]

        np_current_state = np.copy(input_current.numpy())
        np_current_state = np_current_state[0, ...]   # Here I have walls, player and goals for a trajectory frame

        upd_ind = -predict_steps+i
        np_input_traj = input_traj.numpy()
        np_input_traj[0, upd_ind, ...] = np.concatenate((np_current_state, action_map), axis=-1)
        input_traj = tf.convert_to_tensor(np_input_traj, dtype=tf.float32)

        # --------------------------------------------------------
        # 4.3 Update Current State
        # --------------------------------------------------------

        new_current_state = np.concatenate((np_wall_map, new_player_map, np_goal_map), axis=-1)
        new_current_state = tf.convert_to_tensor(new_current_state, dtype=tf.float32)
        new_current_state = tf.expand_dims(new_current_state, axis=0)

        input_current = new_current_state

    coordinates_df = pd.DataFrame(coordinates)
    coordinates_df.to_csv(path_to_save + str("predicted_traj.csv"))

    return predicted_actions

if __name__ == "__main__":

    # To fix ERROR
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ### Load data
    X_train_traj, X_train_current, Y_act_Train = load_training_games(directory=TRAINING_GAMES_PATH,
                                                                     load_percentage=LOAD_PERCENTAGE)
    ### Train the model
    train_model(X_train_traj, X_train_current, Y_act_Train)

    ### Use trained model
    # model =  load_model()
    #
    # ### Keep training the model
    # # history = model.fit(x=X_Train, y=Y_act_Train,
    # #                 epochs=EPOCHS, batch_size=1, verbose=2)
    # # plot_history(history)
    #
    # ### Test it on one prediction
    # X_train_traj, X_train_current, Y_act_Train = load_one_game(directory=TESTING_GAME_PATH) # To test I load one single game
    #
    # # Pick the longest trajectory, which has 14 moves and 15tf frame is current state
    # input_data_traj = X_train_traj[-6, ...]
    # input_data_current = X_train_current[-6, ...]
    # input_data_traj = tf.expand_dims(input_data_traj, axis=0)  # Add axis for "batch_size"
    # input_data_current = tf.expand_dims(input_data_current, axis=0)  # Add axis for "batch_size"
    # actual_action = Y_act_Train[-6]
    # yhat = model.predict([input_data_traj, input_data_current])
    #
    # print("Testing prediction:")
    # print("Actual action: ", actual_action)
    # print("Predicted action: ", yhat)
    #
    # ### Predict trajectory
    # print("Predict Trajectory:")
    # predict_game(model, input_data_traj, predict_steps=5)

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

