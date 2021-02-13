import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import datetime
import pandas as pd
import datetime
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


style.use('ggplot')

# df = pd.read_csv("Unbroken_dataset01.csv")
df = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")
df.columns = ["Datetime", "Total_Feeder"]
df['Datetime'] = pd.to_datetime(df['Datetime'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df['year'] = df['Datetime'].dt.year
df['month'] = df['Datetime'].dt.month
df['day'] = df['Datetime'].dt.day
df['hour'] = df['Datetime'].dt.hour
df['minute'] = df['Datetime'].dt.minute
df['second'] = df['Datetime'].dt.second


df.fillna(value=-99999, inplace=True)
# df = df.set_index('Datetime')
# print(df.dtypes)
# o = df.loc['2013-01-01']
# print(o)
print(df.head())


X = np.array(df['Datetime'])
y = np.array(df['Total_Feeder'])

# print(X[:3])
# print(y[:3])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)