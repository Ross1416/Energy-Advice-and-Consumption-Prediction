import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import datetime
import pandas as pd
import datetime
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit


#https://medium.com/@stallonejacob/time-series-forecast-a-basic-introduction-using-python-414fcb963000

style.use('ggplot')

# df = pd.read_csv("Unbroken_dataset01.csv")
df = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")

df.columns = ["Datetime", "Total_Feeder"]
df['Datetime'] = pd.to_datetime(df['Datetime'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

# df['year'] = df['Datetime'].dt.year
# df['month'] = df['Datetime'].dt.month
# df['day'] = df['Datetime'].dt.day
# df['hour'] = df['Datetime'].dt.hour
# df['minute'] = df['Datetime'].dt.minute

df.set_index('Datetime', inplace=True)
print(df.shape)
df = df.resample('D').mean()
print(df.shape)

df['Yesterday_Feeder'] = df['Total_Feeder'].shift()
df['Diff_Feeder'] = df['Yesterday_Feeder'] - df['Total_Feeder']
df.dropna(inplace=True)
print(df.shape)
# df.fillna(value=-99999, inplace=True)
# print(df.shape)



# print(df.head(10))
# X = np.array(df.drop(['Total_Feeder', 'Datetime'], axis=1))
X = np.array(df.drop(['Total_Feeder'], axis=1))
# y = np.array(df['Total_Feeder'])
y = np.array(df['Total_Feeder'])
X_end = X[-30:]

# print(X[:5])
# print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)

clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

actual = y[-30:]
prediction = clf.predict(X_end)

new_df = df.tail(30)
new_df.insert(0,'Prediction', prediction, True)

new_df.plot(y=['Prediction','Total_Feeder'])
plt.show()


