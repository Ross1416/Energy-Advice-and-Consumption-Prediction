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

df["Month"] = df.index.month
df["DayofWeek"] = df.index.dayofweek
df["DayofMonth"] = df.index.day
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

df.dropna(inplace=True)

# day_df = df.loc["2014-08-08"]
# X = day_df.drop(['Total_Feeder'], axis=1).values
# y = day_df['Total_Feeder'].values

X = []
# future = pd.date_range(start='2016/1/1', end='2016/1/8', freq='10T')
future = pd.date_range(start='2016/3/1', end='2016/3/29', freq='10T')

pd.set_option('display.max_columns', None)

# for i in range(len(future)):
#     X.append([future[i].year, future[i].month, future[i].day, future[i].dayofyear, future[i].weekofyear, future[i].hour,
#               future[i].minute])

for i in range(len(future)):
    X.append([future[i].month, future[i].dayofweek, future[i].day,future[i].hour,future[i].minute])

X = np.array(X)
# df_future = pd.DataFrame(X, columns=['Year', 'Month', 'Day', 'DayofYear', 'Week', 'Hour', 'Minute'], index=future)
df_future = pd.DataFrame(X, columns=['Month', 'DayofWeek', 'DayofMonth', 'Hour', 'Minute'], index=future)


# model_file = "model.pickle"
model_file = "fullmodel_noYear_noWeek.pickle"

with open(model_file, 'rb') as file:
    svr = pickle.load(file)

predictions = svr.predict(X)

df_future['Predictions'] = predictions
df_future['Predictions'].plot()
plt.show()

print(df_future.head())
print(df_future.tail())
