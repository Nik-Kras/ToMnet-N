import Environment
import Agent
import numpy as np

ROWS = 12
COLS = 12
SIGHT = 3

env = Environment.GridWorld(tot_row=ROWS, tot_col=COLS)
env.reset()

# Create a Map for test
# walls = np.ones((ROWS, COLS))
# for i in range(ROWS):
#     if i%2 == 0:
#         walls[i, :] = 0
# env.setStateMatrix(walls, set="walls")
# env.setPosition()
# env.render()

agent = Agent.AgentRL(env, SIGHT)

while True:
    agent.update_world_observation()
    agent.render()

    action = agent.chose_action()
    print(action)

    observe, terminate, goal_picked, reward = env.execute(action)

    if goal_picked:
        print("You have picked a goal, reward = {}".format(reward))
        agent.on_pickup(reward)

    if terminate:
        print("Game result: ", reward)
        break

    #input("Press the <Enter> key to continue...")

env.render()