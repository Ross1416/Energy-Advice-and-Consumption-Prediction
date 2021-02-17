import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
import numpy as np

## https://realpython.com/linear-regression-in-python/
## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

df = pd.read_csv("Unbroken_dataset01.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

# df["DayOfYear"] = df.index.dayofyear
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute
# df["Day"] = df.index.dayofweek

df = df.loc["2015-03-13"]
# df = df.loc["2015-03-23":"2015-03-29"]

# Other option ups accuracy from 50% to 62%
# start_min = df.iloc[0]["Minute"]
#
# mins_from_start = []
# for min in df["Minute"].values:
#     total_min = int(min - start_min)
#     mins_from_start.append(total_min)
#
# df["MinFromStart"] = mins_from_start

X = df.drop(['Total_Feeder'], axis=1).values
y = df['Total_Feeder'].values

# print(df.head())
# print(X[:5])
# print(y[:5])
# print(df.shape)
# print(X.shape)
# print(y.shape)

train_split = 0.9
num_train = int(len(X) * 0.9)
X_train = X[:num_train]
X_test = X[num_train:]

y_train = y[:num_train]
y_test = y[num_train:]


############### all crap as well
# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k, confidence)
#     predictions = clf.predict(X)
#     df['Prediction'] = predictions
#     df['Total_Feeder'].plot()
#     df['Prediction'].plot()
#     plt.show()


#################### 1st Try shite
# clf = LinearRegression()
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)
# predictions = clf.predict(X)
#######################


################## 2nd Try sorta better?
# polynomial_features = PolynomialFeatures(degree=5)  #,include_bias=False)
# linear_regression = LinearRegression()
# pipeline = Pipeline([("polynomial_features", polynomial_features),
#                          ("linear_regression", linear_regression)])
#
# pipeline.fit(X_train,y_train)
# predictions = pipeline.predict(X)
######################

################# 3rd try ####### SEEMS TO WORK
svr = SVR(kernel='rbf', C=40, gamma='auto')

svr.fit(X_train,y_train)
accuracy = svr.score(X_test,y_test)
print(accuracy)             ## only 50% accuracy tho
predictions = svr.predict(X)
###############


df['Prediction'] = predictions
df['Total_Feeder'].plot()
df['Prediction'].plot()
plt.show()