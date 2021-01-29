import pandas as pd

data = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")

data.columns = ["Date_time", "Total_Feeder"]

print("First 2 lines:")
print(data.head(2))
