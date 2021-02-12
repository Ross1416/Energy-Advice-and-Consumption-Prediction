import pandas as pd


# FUNCTION TO FIND LARGEST UNBROKEN DATASETS
# fullDatasetPath: the path to the full dataset
# outputPath: the path to where you want the ouputted csv to go
def findData(fullDatasetPath, outputPath):
    df = pd.read_csv(fullDatasetPath)

    df.columns = ["Date_time", "Total_feeder"]
    df['Date_time'] = pd.to_datetime(df['Date_time'], format='%Y-%m-%d %H:%M:%S')

    chunkFound = False
    tmp_start = 0
    start = 0
    end = 0
    longestChunk = 0

    for index, row in df.iterrows():
        if pd.notna(row["Total_feeder"]):
            if not chunkFound:
                tmp_start = index
                chunkFound = True
        else:
            if chunkFound:
                length = index - tmp_start

                if length > longestChunk:
                    longestChunk = length
                    start = tmp_start
                    end = index

                chunkFound = False

        # Testing purposes - To limit how much data is iterated through
        # count += 1
        # if count > 10000:
        #     break

    # Find the start and end date time from the index
    startDatetime = df.iloc[start]["Date_time"]
    endDatetime = df.iloc[end]["Date_time"]

    print("Longest unbroken is", longestChunk)  # 4032
    print("Starting at", startDatetime, start)  # 2015-03-13 00:01:00 115344
    print("Ending at", endDatetime, end)  # 2015-04-10 00:01:00 119376

    outputData = df[start:end]
    outputData.to_csv(outputPath, index=False)


findData("../Energy_Advice_and_Consumption_Prediction_Dataset.csv", "trainingDataset.csv")
