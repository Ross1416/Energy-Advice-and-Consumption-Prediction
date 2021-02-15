import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

df = pd.read_csv("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["DayOfYear"] = df.index.dayofyear
df["Week"] = df.index.week
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

# print(df.head())

df.dropna(inplace=True)

X = df.drop(['Total_Feeder'], axis=1).values
y = df['Total_Feeder'].values


train_split = 0.9
num_train = int(len(X) * 0.9)
X_train = X[:num_train]
X_test = X[num_train:]

y_train = y[:num_train]
y_test = y[num_train:]


svr = SVR(kernel='rbf', C=40, gamma='auto')

svr.fit(X_train,y_train)
accuracy = svr.score(X_test,y_test)
print(accuracy)     ## only 67%

model_file = "model.pickle"
with open(model_file, 'wb') as file:
    pickle.dump(svr, file)

# Load from file
# with open(model_file, 'rb') as file:
#     pickle_model = pickle.load(file)

predictions = svr.predict(X)

# df['Prediction'] = predictions
# df['Total_Feeder'].plot()
# df['Prediction'].plot()
# plt.show()

