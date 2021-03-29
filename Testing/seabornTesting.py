import seaborn
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("trainingDataset.csv")  # Read unbroken CSV data file

data.columns = ["Date_time", "Total_Feeder"]  # Rename unnamed column to Data_time

data['Date_time'] = pd.to_datetime(data['Date_time'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

seaborn.displot(data['Total_Feeder'])
plt.show()