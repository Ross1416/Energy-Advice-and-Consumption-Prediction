import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("Unbroken_dataset01.csv", )

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
# df["DayOfYear"] = df.index.dayofyear      ## Up to 20% from 12% when i remove this
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

X = df.drop(['Total_Feeder'], axis=1).values
y = df['Total_Feeder'].values

tscv = TimeSeriesSplit()

for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# train_split = 0.9
# num_train = int(len(X) * 0.9)
# X_train = X[:num_train]
# X_test = X[num_train:]
#
# y_train = y[:num_train]
# y_test = y[num_train:]

svr = SVR(kernel='rbf', C=40, gamma='auto')

svr.fit(X_train,y_train)
accuracy = svr.score(X_test,y_test)
print(accuracy)     ## only 20%
predictions = svr.predict(X)

df['Prediction'] = predictions
df['Total_Feeder'].plot()
df['Prediction'].plot()
plt.show()

