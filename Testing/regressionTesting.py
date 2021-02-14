import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# df = pd.read_csv("Unbroken_dataset01.csv")
df = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")


df.columns = ["Datetime", "Total_Feeder"]
df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

df["DayOfYear"] = df.index.dayofyear
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

# print(df.head())
df.dropna(inplace=True)
# print(df.head())

# df = df.resample('1H').mean()
X = df.drop(['Total_Feeder'], axis=1).values
y = df['Total_Feeder'].values

# print(X[:5], X.shape)
# print(y[:5], y.shape)


# SPLIT TRAIN TEST DATA
train_split = 0.9
num_train = int(len(X) * 0.9)
X_train = X[:num_train]
X_test = X[num_train:]

y_train = y[:num_train]
y_test = y[num_train:]

# X_train, y_train, X_test, y_test = TimeSeriesSplit(X, y, 0.1)

# print(X_train.shape, X_test.shape)
# print(len(X_train)+len(X_test))

# print(np.min(X_train))
# print(np.max(X_train))


# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k, confidence)
#     predictions = clf.predict(X)
#     df['Prediction'] = predictions
#
#     df.plot()
#     plt.show()

clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
predictions = clf.predict(X)
df['Prediction'] = predictions

df['Total_Feeder'].plot()
df['Prediction'].plot()

plt.show()
