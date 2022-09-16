import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("grid_search.csv")

data1 = [df["Agent 1 Rate"][:-4], df["Agent 2 Rate"][:-4], df["Agent 3 Rate"][:-4], df["Agent 4 Rate"][:-4]]
data1 = [x.rolling(window=10).mean() for x in data1]

plt.figure()
plt.plot(data1[0])
plt.plot(data1[1])
plt.plot(data1[2])
plt.plot(data1[3])
#plt.show()

data2 = [df["Agent 5 Rate"][:-4], df["Agent 6 Rate"][:-4], df["Agent 7 Rate"][:-4], df["Agent 8 Rate"][:-4]]
data2 = [x.rolling(window=10).mean() for x in data2]

plt.figure()
plt.plot(data2[0])
plt.plot(data2[1])
plt.plot(data2[2])
plt.plot(data2[3])
#plt.show()

data3 = [df["Agent 9 Rate"][:-4], df["Agent 10 Rate"][:-4], df["Agent 11 Rate"][:-4], df["Agent 12 Rate"][:-4]]
data3 = [x.rolling(window=10).mean() for x in data3]

plt.figure()
plt.plot(data3[0])
plt.plot(data3[1])
plt.plot(data3[2])
plt.plot(data3[3])
plt.show()

# print(data1[0])
#
# mean = [x.mean() for x in data1]
#
# print(mean)
#
# plt.plot(data1[0])
# plt.show()

# plt.figure()
# plt.plot(df["Agent 1 Rate"][:-4])
# plt.plot(df["Agent 2 Rate"][:-4])
# plt.plot(df["Agent 3 Rate"][:-4])
# plt.plot(df["Agent 4 Rate"][:-4])
# plt.show()