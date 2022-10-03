import Environment

ROWS = 12
COLS = 12

env = Environment.GridWorld(tot_row=ROWS, tot_col=COLS, consume_goals=1, shaffle=False)
env.reset()
env.render()
env.draw_map()