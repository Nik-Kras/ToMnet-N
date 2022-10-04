import Environment

ROWS = 21
COLS = 21

env = Environment.GridWorld(tot_row=ROWS, tot_col=COLS, consume_goals=1, shaffle=False)
env.reset()
env.render()
env.draw_map()




