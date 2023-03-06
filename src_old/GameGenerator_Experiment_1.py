import Environment
import Agent
import numpy as np

ROWS = 12
COLS = 12
SIGHT = 24

for i in range(4000):

    env = Environment.GridWorld(tot_row=ROWS, tot_col=COLS, consume_goals=1, shaffle=False)
    # Create a Map for test
    walls = np.ones((ROWS, COLS))
    # for i in range(ROWS):
    #     if i%2 == 0:
    #         walls[i, :] = 0
    env.setStateMatrix(walls, set="walls")
    env.setPosition()
    env.save_initial_map()

    # env.reset()

    agent = Agent.AgentStar(env, SIGHT, observability="full")

    while True:
        agent.update_world_observation()
        agent.render()

        action = agent.chose_action(observability="full")
        print(action)

        observe, terminate, goal_picked, reward = env.execute(action)

        if goal_picked:
            print("You have picked a goal, reward = {}".format(reward))
            agent.on_pickup(reward)

        if terminate:
            print("Game result: ", reward)
            break

    agent.save_game(name="Experiment 1")