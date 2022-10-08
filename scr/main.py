
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

def map_from_traj(traj):


    map_1 = traj

    return map_1

if __name__ == "__main__":

    # To fix ERROR: OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
    # OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # SHOULD DELETE?

    # --------------------------------------------------------
    # CONSTANTS and Parameters
    # --------------------------------------------------------
    BATCH_SIZE = 16
    ROW = 12
    COL = 12
    DEPTH = 10
    MAX_TRAJ = 15

    Accuracies = []
    TrainingAccuracy = pd.DataFrame()
    for i in range(1):
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
            data_handler.load_all_games(directory=path_exper_1,
                                        use_percentage=0.25)

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

        # data_processor.validate_data(Data)

        Data = data_processor.zero_padding(max_elements= MAX_TRAJ,
                                                  DictData=Data)

        DataProcessed = data_processor.unite_traj_current(Data)

        # Make Tensors from List
        X_Train, X_Test, X_Valid, \
        Y_goal_Train, Y_goal_Test, Y_goal_Valid, \
        Y_act_Train, Y_act_Test, Y_act_Valid = dict_to_tensors(DataProcessed)

        # --------------------------------------------------------
        # 3. Create and set the model
        # --------------------------------------------------------
        print("----")
        print("Create a model")
        t = ToMnet.ToMnet(ts = MAX_TRAJ,
                          w = ROW,
                          h = COL,
                          d = DEPTH)
        t.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(), # tf.keras.optimizers.Adam(learning_rate=0.0001)
                  metrics=['accuracy'])

        # t.fit(x=X_Train, y=Y_act_Train, validation_data=(X_Valid, Y_act_Valid),
        #       epochs=1, batch_size=16, verbose=2)
        #
        # t.summary()

        # --------------------------------------------------------
        # 4. Train the model
        # --------------------------------------------------------
        print("Train a Model")
        Amplitude = 4.663 / 100000000
        Tau = 33
        N_EPOCHS = 100
        history = t.fit(x=X_Train, y=Y_act_Train, validation_data=(X_Valid, Y_act_Valid),
              epochs=N_EPOCHS, batch_size=16, verbose=2,
                        callbacks=[
                            tf.keras.callbacks.LearningRateScheduler(
                                lambda epoch: Amplitude * 10 ** (epoch / Tau)
                            )
                        ]
                        )

        TrainingAccuracy = TrainingAccuracy.append(pd.DataFrame({str(i): history.history["accuracy"]}))
        print("TrainingAccuracy", TrainingAccuracy)

        plt.plot(
            np.arange(1, N_EPOCHS+1),
            history.history['loss'],
            label='Loss', lw=3
        )
        plt.plot(
            np.arange(1, N_EPOCHS+1),
            history.history['val_accuracy'],
            label='Accuracy', lw=3
        )
        plt.plot(
            np.arange(1, N_EPOCHS+1),
            history.history['val_loss'],
            label='Loss', lw=3
        )
        plt.plot(
            np.arange(1, N_EPOCHS+1),
            history.history['accuracy'],
            label='Accuracy', lw=3
        )
        plt.plot(
            np.arange(1, N_EPOCHS+1),
            history.history['lr'],
            label='Learning rate', color='#000', lw=3, linestyle='--'
        )
        plt.title('Evaluation metrics', size=20)
        plt.xlabel('Epoch', size=14)
        plt.legend()
        plt.show()

        # --------------------------------------------------------
        # 5. Evaluate the model
        # --------------------------------------------------------
        _, accuracy = t.evaluate(x=X_Test, y=Y_act_Test)
        print('Accuracy: %.2f' % (accuracy * 100))

        Accuracies.append(accuracy * 100)

        # # summarize history for accuracy
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        # --------------------------------------------------------
        # 6. Predict with ToMnet-N
        # --------------------------------------------------------

        # single_game = data_handler.load_one_game(path_exper_1)
        # single_game = data_processor.zero_pad_single_game(MAX_TRAJ, single_game)
        # init_map = map_from_traj(single_game[0])

        # --------------------------------------------------------
        # 7. Save the model
        # --------------------------------------------------------


    TrainingAccuracy.to_csv('TrainingAccuracy.csv')

    print("Accuracies: ", Accuracies)

    print("------------------------------------")
    print("Congratultions! You have reached the end of the script.")
    print("------------------------------------")

