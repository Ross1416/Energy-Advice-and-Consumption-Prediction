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
def GetPredictions(model, date_from, date_to, save=False, save_folder=None):
    date_from = pd.to_datetime(date_from, format='%d/%m/%Y')
    date_to = pd.to_datetime(date_to, format='%d/%m/%Y')

    date_range = pd.date_range(start=date_from, end=date_to, freq='10T')

    df = pd.DataFrame({'Month': date_range.month, 'DayofWeek': date_range.dayofweek, "DayofMonth": date_range.day, "Hour": date_range.hour, "Minute": date_range.minute}, index=date_range)
    X = df.values.tolist()

    predictions = model.predict(X)
    df['Predictions'] = predictions

    if save:
        date_from = str(date_from).replace(" ", ".").replace(":", "-")
        date_to = str(date_to).replace(" ", ".").replace(":", "-")

        if save_folder == None:
            save_path = date_from + ";" + date_to + ".csv"
        else:
            save_path = save_folder + "\\" + date_from + ";" + date_to + ".csv"

        df['Predictions'].to_csv(save_path)

    return df['Predictions'].to_frame()


# PLOTS PREDICTIONS ON GRAPH
def Plot(predictions, save=False, save_folder=None, actuals=None):
    ax = predictions.plot()

    if actuals is not None:
        try:
            actuals = actuals.loc[predictions.first_valid_index():predictions.last_valid_index()]
            actuals.columns = ["Actual"]
            actuals.plot(ax=ax)
        except:
            print("The date time range is not within the dataset.")

    plt.ylabel("Energy Consumption (kW)")
    plt.xlabel("Date and time")
    plt.title("Energy Consumption against Date and time")
    plt.tight_layout()
    plt.legend()

    if save:
        date_from = str(predictions.first_valid_index()).replace(" ", ".").replace(":", "-")
        date_to = str(predictions.last_valid_index()).replace(" ", ".").replace(":", "-")

        if save_folder == None:
            save_path = date_from + ";" + date_to + ".jpg"
        else:
            save_path = save_folder + "\\" + date_from + ";" + date_to + ".jpg"

        plt.savefig(save_path)

    plt.show()


# model = GetModel("Models/fullmodel_noYear_noWeek.pickle")
# df = GetPredictions(model, "02/03/2021", "03/03/2021", save=True, save_folder="Predictions")
#
# Plot(df, save=True, save_folder="Predictions")
