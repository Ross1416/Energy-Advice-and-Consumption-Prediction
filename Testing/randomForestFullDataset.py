import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

X = df.values
imputer = KNNImputer(n_neighbors=2)
X = imputer.fit_transform(X)

# df['Total_Feeder'] = X
# print(df.head())

# df["Year"] = df.index.year
# df["Month"] = df.index.month
# df["Day"] = df.index.day
# df["DayOfWeek"] = df.index.dayofweek
# # df["DayOfYear"] = df.index.dayofyear
# df["Week"] = df.index.week
# df["Hour"] = df.index.hour
# df["Minute"] = df.index.minute
#
# X = df.drop(['Total_Feeder'], axis=1).values
# y = df['Total_Feeder'].values
#
# train_split = 0.9
# num_train = int(len(X) * 0.9)
# X_train = X[:num_train]
# X_test = X[num_train:]
#
# y_train = y[:num_train]
# y_test = y[num_train:]
#
# # reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=20)
# reg = KNeighborsRegressor(n_neighbors=2)
#
# reg.fit(X_train, y_train)
# accuracy = reg.score(X_test, y_test)
# print(accuracy)
#
# predictions = reg.predict(X)
#
# df['Prediction'] = predictions
# df['Total_Feeder'].plot()
# df['Prediction'].plot()
# plt.show()