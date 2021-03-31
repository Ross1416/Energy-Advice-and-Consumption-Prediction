from Main import *

# Get data from CSV
data = GetData("Energy_Advice_and_Consumption_Prediction_Dataset.csv")

# Get the data with null energy values from the dataset
df_nans = data[data.isna().all(axis=1)].copy()

# Split the datetime into multiple individual columns
df_nans["Month"] = df_nans.index.month
df_nans["DayofWeek"] = df_nans.index.dayofweek
df_nans["DayofMonth"] = df_nans.index.day
df_nans["Hour"] = df_nans.index.hour
df_nans["Minute"] = df_nans.index.minute

X = df_nans.drop(['Total_Feeder'], axis=1).values

model = GetModel("Models/KNN.pickle")

# Create and store predictions
predictions = model.predict(X)
df_nans['Prediction'] = predictions

# Plot original dataset
data['Total_Feeder'].plot()
plt.ylabel("Energy Consumption (kW)")
plt.xlabel("Date and time")
plt.title("Energy Consumption against Date and time for entire dataset")
plt.tight_layout()
plt.legend()
plt.show()

# Plot of predictions and original dataset alongside each other
df_nans['Prediction'].plot(color="#ff7f0e")
data['Total_Feeder'].plot(color="#1f77b4")

plt.ylabel("Energy Consumption (kW)")
plt.xlabel("Date and time")
plt.title("Energy Consumption against Date and time for entire dataset")
plt.tight_layout()
plt.legend()
plt.show()

# Fill in the null values with the predictions
predictions_df = df_nans['Prediction'].to_frame()
predictions_df.columns = ["Total_Feeder"]

data.fillna(value=predictions_df, inplace=True)

# Plot the new filled dataset
data.plot()
plt.ylabel("Energy Consumption (kW)")
plt.xlabel("Date and time")
plt.title("Energy Consumption against Date and time for entire dataset")
plt.tight_layout()
plt.legend()
plt.show()
