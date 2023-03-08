import pandas as pd
import matplotlib.pyplot as plt

def draw_map(map: pd.DataFrame):
    plt.axis('off')
    plt.imshow(map.to_numpy(dtype=float))
    plt.show()