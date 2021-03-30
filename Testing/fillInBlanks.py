import pandas as pd
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)

df_nans = df[df.isna().all(axis=1)]

# Split the datetime into multiple individual columns
# df_nans["Year"] = df_nans.index.year
df_nans["Month"] = df_nans.index.month
df_nans["DayofWeek"] = df_nans.index.dayofweek
df_nans["DayofMonth"] = df_nans.index.day
df_nans["Hour"] = df_nans.index.hour
df_nans["Minute"] = df_nans.index.minute


X = df_nans.drop(['Total_Feeder'], axis=1).values


with open("../Models/old/fullmodel_noYear_noWeek.pickle", 'rb') as file:
    model = pickle.load(file)

predictions = model.predict(X)
df_nans['Prediction'] = predictions

df['Total_Feeder'].plot()
plt.legend()
plt.show()

df_nans['Prediction'].plot(color="#ff7f0e")
df['Total_Feeder'].plot(color="#1f77b4")
plt.legend()
plt.show()

# p = pd.Series(predictions)
df.fillna(axis=1, value=df_nans['Prediction'])
plt.legend()
df.plot()
plt.show()
