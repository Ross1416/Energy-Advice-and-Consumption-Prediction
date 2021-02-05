import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../Energy_Advice_and_Consumption_Prediction_Dataset.csv")  # Read CSV data file

data.columns = ["Date_time", "Total_Feeder"]  # Rename unnamed column to Data_time

data['Date_time'] = pd.to_datetime(data['Date_time'],
                                   format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

chunkFound = False

tmp_start = 0
tmp_end = 0

startIndex = 0
endIndex = 0
longestUnbroken = 0

count = 0


for index, row in data.iterrows():
    if pd.notna(row["Total_Feeder"]):

        if not chunkFound:
            tmp_start = index
            chunkFound = True



    else:
        if chunkFound:
            tmp_end = index
            # print(tmp_start,tmp_end)
            lengthUnbroken = tmp_end - tmp_start
            # print(lengthUnbroken)
            if lengthUnbroken > longestUnbroken:
                startIndex = tmp_start
                endIndex = tmp_end
                longestUnbroken = lengthUnbroken
            chunkFound = False

    count += 1
    if count > 10000:
        break


startDatetime = data.iloc[startIndex]["Date_time"]
endDatetime = data.iloc[endIndex]["Date_time"]

print("Longest unbroken is ", longestUnbroken)
print("Starting at ", startDatetime, startIndex)
print("Ending at ", endDatetime, endIndex)
