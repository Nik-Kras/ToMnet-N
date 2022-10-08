import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

X = 4.73944 / 1000000
t = 43.024

x = np.array(list(range(1, 101))) # 1-100 epochs
y = X * pow(10, x/t)

plt.plot(y)
plt.show()
