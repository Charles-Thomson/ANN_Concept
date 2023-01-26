import pandas as pd
from matplotlib import pyplot as plt
import csv


# Layout
plt.rcParams["figure.figsize"] = [20.0, 10.0]
plt.rcParams["figure.autolayout"] = True
columns = ["Episode", "Reward", "Threshold"]
columns_b = ["Time", "Threshold"]


plt.xlabel("Time")
plt.ylabel("Fitness")
plt.xticks(range(1, 1000, 50))
# plt.yticks(range(1, 110, 10))

df = pd.read_csv("test.csv", usecols=columns)


print("Contents in csv file:", df)
(plot1,) = plt.plot(df.Episode, df.Reward)
(plot2,) = plt.plot(df.Episode, df.Threshold)
plt.legend([plot1, plot2], ["Fitness", "Threshold"], loc="best")
plt.show()
