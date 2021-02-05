import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")  # Read CSV data file

data.columns = ["Date_time", "Total_feeder"]  # Rename unnamed column to Data_time

data['Date_time'] = pd.to_datetime(data['Date_time'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

chunkFound = False

tmp_start = 0
tmp_end = 0

startIndex = 0
endIndex = 0
longestUnbroken = 0

# count = 0 # Testing purposes

for index, row in data.iterrows():

    if pd.notna(row["Total_feeder"]):

        if not chunkFound:
            tmp_start = index
            chunkFound = True
    else:
        if chunkFound:
            tmp_end = index
            lengthUnbroken = tmp_end - tmp_start

            if lengthUnbroken > longestUnbroken:
                startIndex = tmp_start
                endIndex = tmp_end
                longestUnbroken = lengthUnbroken
            chunkFound = False

    # Testing purposes - To limit how much data is iterated through
    # count += 1
    # if count > 10000:
    #     break

# Find the start and end date time from the index
startDatetime = data.iloc[startIndex]["Date_time"]
endDatetime = data.iloc[endIndex]["Date_time"]

print("Longest unbroken is", longestUnbroken)  # 4032
print("Starting at", startDatetime, startIndex)  # 2015-03-13 00:01:00 115344
print("Ending at", endDatetime, endIndex)  # 2015-04-10 00:01:00 119376

unbroken_dataset = data[startIndex:endIndex]
# unbroken_dataset.to_csv("Unbroken_dataset01.csv", index=False)
