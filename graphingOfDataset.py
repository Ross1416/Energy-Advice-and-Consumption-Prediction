import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Energy_Advice_and_Consumption_Prediction_Dataset.csv")    # Open the CSV file as a pandas Dataframe

print("INFO:")
print(df.info())

print("DESCRITION:")
print(df.describe())

print("HEAD:")
print(df.head())        # Print the first 5 rows of the data

print("TAIL:")
print(df.tail())        # Print the last 5 rows of the data

print("SHAPE:")
print(df.shape)         # Print the shape of the data


df.columns = ["Datetime", "Total_Feeder"]       # Rename the column headings so that the unamed one becomes 'Datetime'

print("DATATYPES:")
print(df.dtypes)        # Print the types of data in each column

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')  # changes the 'Datetime' column to a datetime data type

print("DATATYPES:")
print(df.dtypes)        # Print the types of data in each column

df.set_index('Datetime', inplace=True)      # Replace the numeric index with the 'Datetime' column
#
# PLOT OF ENTIRE DATASET
df.plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Date and time")
plt.title("Energy Consumption against Date and time")
plt.tight_layout()
plt.show()


# PLOT OF SINGLE DAYS
day_df = df.loc["2015-03-01"]
day_df["Time"] = day_df.index.time
day_df['Time'] = pd.to_datetime(day_df['Time'], format='%H:%M:%S').dt.time
day_df.set_index('Time', inplace=True)

day_df.columns = ['2015-03-01']
day_df['2015-03-02'] = df.loc["2015-03-02"].values
day_df['2015-03-03'] = df.loc["2015-03-03"].values
ax = day_df.plot(x_compat=True)

ax.set_xticks(["00:00","02:00","04:00","06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00"])
ax.set_xlim("00:00", "23:51")


plt.ylabel("Energy Consumption")
plt.xlabel("Time")
plt.title("Energy Consumption against time on individual days")
plt.tight_layout()
plt.show()

# PLOT OF A SINGLE WEEK
df.loc["2015-03-23":"2015-03-29"].plot()
plt.ylabel("Energy Consumption")
plt.xlabel("Day of Week")
plt.title("Energy Consumption against time on the week of the 23/03/2015")
plt.tight_layout()
plt.show()


# PLOT OF A SINGLE MONTH
mean_month = df.loc["2015-09"].resample("D").mean()
mean_month.columns = ["Average consumption of days"]
ax = df.loc["2015-09"].plot()
mean_month.plot(ax=ax)
plt.ylabel("Energy Consumption")
plt.xlabel("Day of Month")
plt.title("Energy Consumption against time in September of 2015")
plt.tight_layout()
plt.show()

# PLOT OF A SINGLE YEAR
mean_year = df.loc["2014"].resample("M").mean()
mean_year.columns = ["Average consumption of months"]
ax = df.loc["2014"].plot()
mean_year.plot(ax=ax)
plt.ylabel("Energy Consumption")
plt.xlabel("Date")
plt.title("Energy Consumption against time in 2014")
plt.tight_layout()
plt.show()


# HISTOGRAM OF ENTIRE DATASET
df.hist(bins=50)
plt.ylabel("Number of occurances")
plt.xlabel("Energy Consumption")
plt.title("Histogram of Energy Consumption for entire dataset")
plt.tight_layout()
plt.show()