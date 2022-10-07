import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Accuracy with CNN kernel initialisation with He distribution
# And with default Xavier distribution
Xavier = [53 , 17 , 60 , 19,
49 , 48 , 51 , 24,
14 , 61 , 22 , 28 ]
He = [54 , 50 , 56 , 21 ,
51 , 30 , 24 , 49,
21 , 22 , 35 , 51]

data = {
    "Xavier": Xavier,
    "He": He
}

data = pd.DataFrame(data) # (np.transpose(np.array([Xavier, He])))
penguins = sns.load_dataset("penguins")

# plt.hist(Xavier, bins=10)
# plt.hist(He, bins=10)
# plt.show()

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)
sns.histplot(data, multiple="dodge", shrink=.8, bins=15)
plt.xlim(10,60)
sns.displot(data, color="dodgerblue", kind="kde")
plt.xlim(0,80)
plt.legend()
plt.show()
