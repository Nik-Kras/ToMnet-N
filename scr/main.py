# import tensorforce
import Environment
import Agent
import numpy as np
import matplotlib.pyplot as plt
from MapGenerator.Grid import *
# from tensorforce.execution import Runner

# import pandas as pd
# import tensorflow as tf

SIZE = 4
ROWS = 12
COLS = 12
SIGHT = 3

def main():
    print("Hi, Nikita")

    environment = game # tensorforce.Environment.create(environment=dict(environment='gym', level='CartPole'), max_episode_timesteps=500)
    # agent = tensorforce.Agent.create(agent='ppo', environment=environment, batch_size=10,
    #                                  learning_rate=1e-3, max_episode_timesteps=500)

    # agent = tensorforce.Agent.create(
    #     agent='ppo', environment=environment, max_episode_timesteps=30,
    #     # Automatically configured network
    #     network='auto',
    #     # Optimization
    #     batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
    #     # Reward estimation
    #     likelihood_ratio_clipping=0.2, discount=0.99,
    #     # Exploration
    #     exploration=0.0, variable_noise=0.0,
    #     # Regularization
    #     l2_regularization=0.0, entropy_regularization=0.0,
    #     # TensorFlow etc
    #     parallel_interactions=5
    # )

    # agent = tensorforce.Agent.create(
    #     agent='tensorforce', environment=environment, update=64,
    #     optimizer=dict(optimizer='adam', learning_rate=1e-3),
    #     objective='policy_gradient',
    #
    #     reward_estimation=dict(horizon=15, discount=0.99),    #
    #
    #     policy = dict(
    #         network = dict(
    #             type="auto",
    #             # rnn=15,         # Set the Horizon for LSTM
    #             size = 128,
    #             depth = 4
    #         )
    #     ),
    #
    #     exploration = dict( type='exponential', unit='episodes', num_steps=1000, initial_value=0.99, decay_rate=0.5)
    #
    #     #     (
    #     #     type='linear', unit='episodes', num_steps=1000,
    #     #     initial_value=10, final_value=50
    #     # )
    #
    # )

    # agent = tensorforce.Agent.create(
    #     agent='tensorforce',
    #     environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    #     memory=10000,
    #     update=dict(unit='timesteps', batch_size=64),
    #     optimizer=dict(type='adam', learning_rate=3e-4),
    #     policy=dict(network='auto'),
    #     objective='policy_gradient',
    #     reward_estimation=dict(horizon=20)
    # )



    grid_search = [
        # Setup #1. Checking for random results
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=dict(network='auto')),

        # Setup #2. Checking for random results
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=dict(network='auto')),

        # Setup #3. Checking for random results
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=dict(network='auto')),

        # Setup #4. Checking for random results
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=dict(network='auto')),


        # Setup #5. Checking for random results - With complex Neural Network (DENSE)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                ]
             ]),

        # Setup #6. Checking for random results - With complex Neural Network  (DENSE)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     # dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #7. Checking for random results - With complex Neural Network  (DENSE)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #8. Checking for random results - With complex Neural Network  (DENSE)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='flatten'),
                     dict(type='dense', size=64),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #9. Checking for random results - With complex Neural Network  (CONV2D)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #10. Checking for random results - With complex Neural Network  (CONV2D)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #11. Checking for random results - With complex Neural Network  (CONV2D)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Setup #12. Checking for random results - With complex Neural Network  (CONV2D)
        dict(agent='tensorforce',
             environment=environment,
             update=64,
             # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
             memory=dict(type="recent", capacity=10000),
             optimizer=dict(optimizer='adam',
                            learning_rate=dict(type='linear', unit='episodes', num_steps=10000, initial_value=1e-3,
                                               final_value=1e-5)),
             objective='policy_gradient',
             reward_estimation=dict(horizon=1),
             exploration=dict(type='linear', unit='episodes', num_steps=1000, initial_value=0.99, final_value=0.01),
             policy=[
                 [
                     dict(type='retrieve', tensors=['Player']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Player-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Walls']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Walls-embedding')
                 ],
                 [
                     dict(type='retrieve', tensors=['Goals']),
                     #dict(type='embedding', size=64),
                     dict(type='conv2d', size=64),
                     dict(type='flatten'),
                     dict(type='register', tensor='Goals-embedding')
                 ],
                 [
                     dict(
                         type='retrieve', aggregation='concat',
                         tensors=['Goals-embedding', 'Walls-embedding', 'Player-embedding']
                     ),
                     dict(type='dense', size=64),
                     dict(type='dense', size=64),
                 ]
             ]),

        # Check the best from 4 setups above and make 4 experiments with Neural Network of FC based on the best

        # Setup #5. NN architecture #1

        # Setup #6. NN architecture #2

        # Setup #7. NN architecture #3

        # Setup #8. NN architecture #4

        # Check for the best and put increasing of Exploration for top 3 models with experiments of the range
        # exploration=dict(type='linear', unit='episodes', num_steps=5000, initial_value=0.99, final_value=0.01)

        # Setup #9. For top-1 model. Experiment with exploration #1

        # Setup #10. For top-1 model. Experiment with exploration #2

        # Setup #11. For top-2 model. Experiment with exploration #1

        # Setup #12. For top-2 model. Experiment with exploration #2

        # Setup #13. For top-3 model. Experiment with exploration #1

        # Setup #14. For top-3 model. Experiment with exploration #2

        # Setup #15. Try settings from point 6 of LogBook

        # Setup #16. point 9 from LogBook

        # When finished - Rebuild an environment to increase dimensionality and run it again!

    ]

    df = pd.DataFrame()
    cnt_df = 1
    num_eval_episodes = 500
    num_train_episodes = 10000

    for setting in grid_search:
        agent = tensorforce.Agent.create(
            agent=setting["agent"],
            environment=setting["environment"],
            update=setting["update"],
            reward_estimation=setting["reward_estimation"],
            memory=setting["memory"],
            optimizer=setting["optimizer"],
            objective=setting["objective"],
            exploration=setting["exploration"],
            policy=setting["policy"]
        )
        print(agent.get_architecture())


        tracker = {
            "rewards": [0],
            "picked_goal": [0],
            "window": 50,
            "cnt": 0,  # Keep track on counting "window" elements per one array element
            "array_cnt": 0  # Keep track on array indexes
        }
        tracker["rewards"] = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))
        tracker["picked_goal"] = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))

        for episode in range(num_train_episodes):

            # Episode using act and observe
            states = environment.reset()
            terminal = False
            sum_rewards = 0.0
            num_updates = 0

            # Record episode experience
            episode_states = list()
            episode_internals = list()
            episode_actions = list()
            episode_terminal = list()
            episode_reward = list()


            internals = agent.initial_internals()

            while not terminal:

                # Record for learning
                episode_states.append(states)
                episode_internals.append(internals)
                actions, internals = agent.act(states=states, internals=internals, independent=True)
                episode_actions.append(actions)
                states, terminal, reward = environment.execute(actions=actions)
                episode_terminal.append(terminal)
                episode_reward.append(reward)
                sum_rewards += reward

            # Feed recorded experience to agent
            agent.experience(
                states=episode_states, internals=episode_internals, actions=episode_actions,
                terminal=episode_terminal, reward=episode_reward
            )
            # Perform update
            agent.update()

            tracker["rewards"][tracker["array_cnt"]] += sum_rewards
            if sum_rewards > 0: tracker["picked_goal"][tracker["array_cnt"]] += 1

            # Each "window" iterations count average and go to next array element
            if tracker["cnt"] == tracker["window"]:
                current_index = tracker["array_cnt"]
                tracker["rewards"][current_index] = tracker["rewards"][current_index] / tracker["window"]
                tracker["picked_goal"][current_index] = tracker["picked_goal"][current_index] / tracker["window"]
                tracker["array_cnt"] += 1
                tracker["cnt"] = 0
            else:
                tracker["cnt"] += 1
            print('Episode {}: {}'.format(episode, sum_rewards))

        sum_rewards = 0.0

        for g_cnt in range(num_eval_episodes):
            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            cnt = 0
            while not terminal:
                actions, internals = agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = environment.execute(actions=actions)
                sum_rewards += reward
                print("{}/{}".format(cnt + 1, g_cnt + 1))
                cnt += 1
        print('Mean evaluation return:', )
        agent.close()

        df['Agent ' + str(cnt_df) + " Rewards"] = tracker["rewards"]
        df['Agent ' + str(cnt_df) + " Rate"] = tracker["picked_goal"]
        df['Agent ' + str(cnt_df) + " Test"] = sum_rewards / num_eval_episodes
        cnt_df += 1

    print(df)
    df.to_csv('grid_search.csv')


    """

    agent = tensorforce.Agent.create(
        agent='tensorforce', environment=environment,
        update=64, # dict(unit="episodes", batch_size=64, frequency=0.5, start=1000),
        memory=dict(type="recent", capacity=10000),
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=1),

        # Save an agent each 1000 episodes
        saver=dict(directory="saved_agents", filename="last_agent", frequency=1000, unit="episodes"),
        summarizer=dict(directory="summary_agents", filename="last_agent_summary"),
        recorder=dict(directory="recorded_agents", frequency=1000),
    )

    print(agent.get_architecture())

    ### Checking new format of running
    # agent_cnt = 1
    # runner = Runner(
    #     agent=agent,
    #     environment=environment,
    #     max_episode_timesteps=50
    #     # num_parallel=5, remote='multiprocessing'
    # )
    # runner.run(num_episodes=10000, save_best_agent="saved_agents/Agent_" + str(agent_cnt))
    # runner.run(num_episodes=500, evaluation=True)
    # runner.close()
    ### Checking new format of running

    # Train for 20,000 episodes
    num_train_episodes = 20000
    tracker = {
        "rewards": [0],
        "picked_goal": [0],
        "window": 50,
        "cnt": 0,       # Keep track on counting "window" elements per one array element
        "array_cnt": 0  # Keep track on array indexes
    }
    tracker["rewards"]     = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))
    tracker["picked_goal"] = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))

    for episode in range(num_train_episodes):

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0

        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward

        tracker["rewards"][tracker["array_cnt"]] +=  sum_rewards
        if sum_rewards > 0: tracker["picked_goal"][tracker["array_cnt"]] += 1

        # Each "window" iterations count average and go to next array element
        if tracker["cnt"] == tracker["window"]:
            current_index = tracker["array_cnt"]
            tracker["rewards"][current_index]     = tracker["rewards"][current_index] / tracker["window"]
            tracker["picked_goal"][current_index] = tracker["picked_goal"][current_index] / tracker["window"]
            tracker["array_cnt"] += 1
            tracker["cnt"] = 0
        else:
            tracker["cnt"] += 1
        print('Episode {}: return={} moves = {} updates={}'.format(episode, sum_rewards, tracker["cnt"], num_updates))

    plt.plot(tracker["rewards"][:tracker["array_cnt"]-2])
    plt.show()
    plt.plot(tracker["picked_goal"][:tracker["array_cnt"]-2])
    plt.show()

    # Evaluate for 500 episodes
    sum_rewards = 0.0
    g_cnt = 0
    num_eval_episodes = 500
    for g_cnt in range(num_eval_episodes):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        cnt = 0
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
            print("{}/{}".format(cnt+1, g_cnt+1))
            cnt += 1
    print('Mean evaluation return:', sum_rewards / num_eval_episodes)

    """

    environment.close()


if __name__ == '__main__':

    game = Environment.GridWorld(tot_row=ROWS, tot_col=COLS)

    game.reset()
    game.render()
    print(game.getWorldState())

    main()
