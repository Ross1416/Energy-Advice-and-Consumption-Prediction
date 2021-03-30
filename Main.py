import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pickle
import time


# FITS MODEL TO DATASET AND SAVES IT
def CreateModel(dataset_path, output_path=None, type='KNN'):

    # Create Regressor model
    if type == "KNN":
        model = KNeighborsRegressor(n_neighbors=5)
    elif type == "SVR":
        model = SVR(kernel='rbf', C=40, gamma='auto')
    elif type == "RFR":
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
    else:
        print("Not a valid regression algorithm.")

    # Read in dataset and perform preliminary sorting
    df = pd.read_csv(dataset_path)
    df.columns = ["Datetime", "Total_Feeder"]
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)

    # Split the datetime into multiple individual columns
    # df["Year"] = df.index.year
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

    train_split = 0.9   ##   MIGHT BE LOSING SOME DATA WITH THIS METHOD? better way to do it?### https://medium.com/@alexstrebeck/training-and-testing-machine-learning-models-e1f27dc9b3cb
    num_train = int(len(X) * train_split)
    X_train = X[:num_train]
    X_test = X[num_train:]

    y_train = y[:num_train]
    y_test = y[num_train:]


    start_time = time.time()

    # Fit model to data
    model.fit(X_train, y_train)

    time_taken = time.time() - start_time                       # RBF                           # RFR                       #KNN
    print("Time taken:", time_taken, "seconds")                 # 2402.46537232399 seconds      3.545670509338379           0.391770601272583

    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    print("Training accuracy:", accuracy_train)                  # 0.9447776916752073            0.7372641404547604          0.9403859188290737
    print("Testing accuracy:", accuracy_test)                    # 0.6614105265554282            0.77431329514064            0.8014318330506364

    if output_path is not None:
        # Save the model to external file
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)

    return model


# OPENS SAVED MODEL FILE
def GetModel(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


# MAKE PREDICTIONS FROM FITTED MODEL
def MakePredictions(model, start_date, end_date, save=False, save_folder=None):
    # Convert dates into pandas datetime datatype
    start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date = pd.to_datetime(end_date, format='%d/%m/%Y')

    # Create date range with 10 minute intervals
    date_range = pd.date_range(start=start_date, end=end_date, freq='10T')

    # Create a list from the data range
    df = pd.DataFrame({'Month': date_range.month, 'DayofWeek': date_range.dayofweek, "DayofMonth": date_range.day, "Hour": date_range.hour, "Minute": date_range.minute}, index=date_range)
    X = df.values.tolist()

    # Make predictions on the datarange
    predictions = model.predict(X)
    df['Predictions'] = predictions

    # Save the predictions to a .csv file if wanted
    if save:
        start_date = str(start_date).replace(" ", ".").replace(":", "-")
        end_date = str(end_date).replace(" ", ".").replace(":", "-")

        if save_folder == None:
            save_path = start_date + ";" + end_date + ".csv"
        else:
            save_path = save_folder + "\\" + start_date + ";" + end_date + ".csv"

        df['Predictions'].to_csv(save_path)

    return df['Predictions'].to_frame()


# PLOTS PREDICTIONS ON GRAPH
def PlotPredictions(predictions, save=False, save_folder=None, actuals=None):
    # Plot predictions on a graph
    ax = predictions.plot()

    # Plot actual values on graph if actuals is provided and within the date range
    if actuals is not None:
        try:
            actuals = actuals.loc[predictions.first_valid_index():predictions.last_valid_index()]
            actuals.columns = ["Actual"]
            actuals.plot(ax=ax)
        except:
            print("The date time range is not within the dataset.")

    # Format the graph
    plt.ylabel("Energy Consumption (kW)")
    plt.xlabel("Date and time")
    plt.title("Energy Consumption against Date and time")
    plt.tight_layout()
    plt.legend()

    # Save plot to a .jpg file if required
    if save:
        date_from = str(predictions.first_valid_index()).replace(" ", ".").replace(":", "-")
        date_to = str(predictions.last_valid_index()).replace(" ", ".").replace(":", "-")

        if save_folder == None:
            save_path = date_from + ";" + date_to + ".jpg"
        else:
            save_path = save_folder + "\\" + date_from + ";" + date_to + ".jpg"

        plt.savefig(save_path)

    plt.show()


model = CreateModel("Energy_Advice_and_Consumption_Prediction_Dataset.csv", type="SVR") # "Models/RFR_Month_DoW_DoM_Hour_Min.pickle"
# model = GetModel("Models/SVR_Month_DoW_DoM_Hour_Min.pickle")
df = MakePredictions(model, "02/03/2015", "03/03/2015", save=False, save_folder="Predictions")
#
data = pd.read_csv("Energy_Advice_and_Consumption_Prediction_Dataset.csv")
data.columns = ["Datetime", "Total_Feeder"]
data.columns = ["Datetime", "Total_Feeder"]
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')
data.set_index('Datetime', inplace=True)
#
PlotPredictions(df, save=False, save_folder="Predictions", actuals=data)
# # PlotPredictions(df, save=True, save_folder="Predictions")
