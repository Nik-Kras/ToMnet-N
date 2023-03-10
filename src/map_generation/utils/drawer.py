import pandas as pd
import matplotlib.pyplot as plt

def draw_map(map: pd.DataFrame):
    """ Simple map drawer that represents objects as squares """
    plt.axis('off')
    plt.imshow(map.to_numpy(dtype=float))
    plt.show()

### TODO: to be developed
def draw_real_map(map: pd.DataFrame):
    """ Complex map drawer that represents objects with their colors and shapes """
    pass