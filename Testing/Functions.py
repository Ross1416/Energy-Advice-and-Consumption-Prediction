import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.svm import SVR
import pickle
import time


# FITS MODEL TO DATASET AND SAVES IT
def create_model(dataset_path, output_path):
    # Read in dataset
    df = pd.read_csv(dataset_path)
    df.columns = ["Datetime", "Total_Feeder"]

    # Changes the Date_time column to a datetime data type
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

    df.set_index('Datetime', inplace=True)

    # Split the datetime into multiple individual columns
    df["Year"] = df.year
    df["Month"] = df.index.month
    df["DayofWeek"] = df.index.dayofweek
    df["DayofMonth"] = df.index.day
    # df["DayOfYear"] = df.index.dayofyear
    # df["Week"] = df.index.week
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute


    # Removes NaN values
    df.dropna(inplace=True)

    X = df.drop(['Total_Feeder'], axis=1).values
    y = df['Total_Feeder'].values

    train_split = 0.9
    num_train = int(len(X) * train_split)
    X_train = X[:num_train]
    X_test = X[num_train:]

    y_train = y[:num_train]
    y_test = y[num_train:]

    # Create regressor model
    model = SVR(kernel='rbf', C=40, gamma='auto')
    # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=20)
    # model = RadiusNeighborsRegressor(radius=4.3)

    start_time = time.time()

    # Fit model to data
    model.fit(X_train, y_train)

    time_taken = time.time() - start_time
    print("Time taken:", time_taken)

    accuracy = model.score(X_test, y_test)
    print(accuracy)  ## only 67%

    # Save the model to external file
    with open(output_path, 'wb') as file:
        pickle.dump(model, file)


# OPENS SAVED MODEL FILE
def get_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model


# GET PREDICTIONS FROM FITTED MODEL
def get_predictions(model, date_from, date_to=None, dataset_path=None, show_actual=False):
    date_range = pd.date_range(start=date_from, end=date_to, freq='10T')

    if show_actual:
        # Read in dataset
        df = pd.read_csv(dataset_path)
        df.columns = ["Datetime", "Total_Feeder"]
        # Changes the Date_time column to a datetime data type
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('Datetime', inplace=True)
        # Split the datetime into multiple individual columns
        df["Year"] = df.index.year
        df["Month"] = df.index.month
        df["DayofWeek"] = df.index.dayofweek
        df["DayofMonth"] = df.index.day
        # df["DayOfYear"] = df.index.dayofyear
        # df["Week"] = df.index.week
        df["Hour"] = df.index.hour
        df["Minute"] = df.index.minute

        # Removes NaN values
        df.dropna(inplace=True)

        try:
            df = df.loc[date_range[0]:date_range[-1]]
            X = df.drop(['Total_Feeder'], axis=1).values
            df['Total_Feeder'].plot()
        except:
            df = pd.DataFrame(
                {'Month': date_range.month, 'DayofWeek': date_range.dayofweek, "DayofMonth": date_range.day,
                 "Hour": date_range.hour, "Minute": date_range.minute}, index=date_range)
            X = df.values.tolist()
    else:
        df = pd.DataFrame({'Month': date_range.month, 'DayofWeek': date_range.dayofweek, "DayofMonth": date_range.day, "Hour": date_range.hour, "Minute": date_range.minute}, index=date_range)
        X = df.values.tolist()

    predictions = model.predict(X)

    df['Predictions'] = predictions
    df['Predictions'].plot()
    plt.legend()
    plt.show()
    save_path = date_from + date_to
    plt.savefig()


# create_model("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", "fullmodel_noYear_noWeek.pickle")
# model = get_model("../Models/fullmodel_noYear_noWeek.pickle")
# create_model("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", "../Models/RNN_Month_DoW_DoM_Hour_Min.pickle")
# model = get_model("../Models/RNN_Month_DoW_DoM_Hour_Min.pickle")
# get_predictions(model, "19/03/2015", "20/03/2015", "..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", False)

create_model("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", "../Models/RNN_Year_Month_DoW_DoM_Hour_Min.pickle")
model = get_model("../Models/RNN_Year_Month_DoW_DoM_Hour_Min.pickle")
get_predictions(model, "19/03/2015", "20/03/2015", "..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", True)