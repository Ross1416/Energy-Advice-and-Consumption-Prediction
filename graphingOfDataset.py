import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Energy_Advice_and_Consumption_Prediction_Dataset.csv")    # Open the CSV file as a pandas Dataframe

print("HEAD:")
print(df.head())        # Print the first 5 rows of the data
print("")
print("TAIL:")
print(df.tail())        # Print the last 5 rows of the data
print("")
print("SHAPE:")
print(df.shape)         # Print the shape of the data
print("")

df.columns = ["Datetime", "Total_Feeder"]       # Rename the column headings so that the unamed one becomes 'Datetime'

print("DATATYPES:")
print(df.dtypes)        # Print the types of data in each column
print("")

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')  # changes the 'Datetime' column to a datetime data type

print("DATATYPES:")
print(df.dtypes)        # Print the types of data in each column
print("")

df.set_index('Datetime', inplace=True)      # Replace the numeric index with the 'Datetime' column

# PLOT OF ENTIRE DATASET
df.plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Date and time")
plt.title("Energy Consumption against Date and time")
plt.tight_layout()
plt.show()

# PLOT OF A SINGLE DAY
df.loc["2015-03-13"].plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Time")
plt.title("Energy Consumption against time on 13/03/2015")
plt.tight_layout()
plt.show()

# PLOT OF A SINGLE WEEK
df.loc["2015-03-01":"2015-03-07"].plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Time")
plt.title("Energy Consumption against time on the week of the 01/03/2015")
plt.tight_layout()
plt.show()


# PLOT OF A SINGLE MONTH
df.loc["2015-03"].plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Time")
plt.title("Energy Consumption against time in March of 2015")
plt.tight_layout()
plt.show()

# HISTOGRAM OF ENTIRE DATASET
df.hist(bins=50)
plt.ylabel("Number of occurances")
plt.xlabel("Energy Consumption")
plt.title("Histogram of Energy Consumption for entire dataset")
plt.tight_layout()
plt.show()