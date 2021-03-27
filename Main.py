import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.svm import SVR
import pickle
import time


# FITS MODEL TO DATASET AND SAVES IT
def CreateModel(dataset_path, output_path, save):
    # Read in dataset and perform preliminary sorting
    df = pd.read_csv(dataset_path)
    df.columns = ["Datetime", "Total_Feeder"]
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

    # Remove NaN values
    df.dropna(inplace=True)

    # Split data
    X = df.drop(['Total_Feeder'], axis=1).values
    y = df['Total_Feeder'].values

    train_split = 0.9
    num_train = int(len(X) * train_split)
    X_train = X[:num_train]
    X_test = X[num_train:]

    y_train = y[:num_train]
    y_test = y[num_train:]

    # Create Regressor model
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

    if save:
        # Save the model to external file
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)

    return model


# OPENS SAVED MODEL FILE
def GetModel(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model


# GET PREDICTIONS FROM FITTED MODEL
def GetPredictions(model, date_from, date_to=None):
    date_range = pd.date_range(start=date_from, end=date_to, freq='10T')

    df = pd.DataFrame({'Month': date_range.month, 'DayofWeek': date_range.dayofweek, "DayofMonth": date_range.day, "Hour": date_range.hour, "Minute": date_range.minute}, index=date_range)
    X = df.values.tolist()

    predictions = model.predict(X)
    df['Predictions'] = predictions
    return df['Predictions']


def Plot(predictions, save=True, save_folder=None, actuals=None, plot_actual=False):
    predictions.plot()
    plt.legend()

    if save:
        start_date = str(predictions.first_valid_index()).replace(" ", ".").replace(":", "-")
        end_date = str(predictions.last_valid_index()).replace(" ", ".").replace(":", "-")

        if save_folder == None:
            save_path = start_date + ";" + end_date + ".jpg"
        else:
            save_path = save_folder + "\\" + start_date + ";" + end_date + ".jpg"

        plt.savefig(save_path)

    plt.show()


model = GetModel("Models/fullmodel_noYear_noWeek.pickle")
df = GetPredictions(model, "02/03/2021", "03/03/2021")
Plot(df, save=True, save_folder="Predictions")
