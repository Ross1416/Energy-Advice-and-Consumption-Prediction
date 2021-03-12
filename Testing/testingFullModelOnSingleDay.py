import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pickle
import time
import numpy as np

df = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)


# df["Year"] = df.index.year
# df["Month"] = df.index.month
# df["Day"] = df.index.day
# df["DayOfYear"] = df.index.dayofyear
# df["Week"] = df.index.week
# df["Hour"] = df.index.hour
# df["Minute"] = df.index.minute

df["Month"] = df.index.month
df["DayofWeek"] = df.index.dayofweek
df["DayofMonth"] = df.index.day
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

# print(df.head())
df.dropna(inplace=True)
# pd.set_option('display.max_colwidth', -1)
# pd.set_option('display.max_columns', None)
# print(df.head())

day_df = df.loc["2014-08-08"]
# day_df = df.loc["2014-08-01":"2014-08-08"]
X = day_df.drop(['Total_Feeder'], axis=1).values
y = day_df['Total_Feeder'].values


train_split = 0.9
num_train = int(len(X) * 0.9)
X_train = X[:num_train]
X_test = X[num_train:]

y_train = y[:num_train]
y_test = y[num_train:]

# model_file = "fullmodel.pickle"
model_file = "fullmodel_noYear_noWeek.pickle"
# model_file = "model.pickle"


with open(model_file, 'rb') as file:
    svr = pickle.load(file)


predictions = svr.predict(X)


day_df['Prediction'] = predictions
day_df['Total_Feeder'].plot()
day_df['Prediction'].plot()
plt.show()

