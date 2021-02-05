import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")  # Read original CSV data file
data = pd.read_csv("Unbroken_dataset01.csv")  # Read unbroken CSV data file

data.columns = ["Date_time", "Total_Feeder"]  # Rename unnamed column to Data_time

data['Date_time'] = pd.to_datetime(data['Date_time'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

print("First 2 lines:")
print(data.head(2))

data.plot(x='Date_time', y='Total_Feeder')  # this plots the graph
plt.show()  # shows the plot

# If this is right then there seems to be a big chunk of data round about july 2014

print(data.shape)
