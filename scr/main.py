import Environment

Environment.createGame()

state = Environment.initGridRand()

state = Environment.makeMove(state, 1)
print(Environment.dispGrid(state))
state = Environment.makeMove(state, 1)
print(Environment.dispGrid(state))
state = Environment.makeMove(state, 1)
print(Environment.dispGrid(state))
state = Environment.makeMove(state, 3)
print(Environment.dispGrid(state))

print('Reward: %s' % (Environment.getReward(state),))
print(Environment.dispGrid(state))