import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pickle
import time

def create_model(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    df.columns = ["Datetime", "Total_Feeder"]

    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

    df.set_index('Datetime', inplace=True)

    df["Month"] = df.index.month
    df["DayofWeek"] = df.index.dayofweek
    df["DayofMonth"] = df.index.day
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute

    df.dropna(inplace=True)

    X = df.drop(['Total_Feeder'], axis=1).values
    y = df['Total_Feeder'].values

    train_split = 0.9
    num_train = int(len(X) * train_splitu)
    X_train = X[:num_train]
    X_test = X[num_train:]

    y_train = y[:num_train]
    y_test = y[num_train:]


    svr = SVR(kernel='rbf', C=40, gamma='auto')

    start_time = time.time()

    svr.fit(X_train, y_train)

    time_taken = time.time() - start_time
    print("Time taken:", time_taken)

    accuracy = svr.score(X_test, y_test)
    print(accuracy)  ## only 67%

    with open(output_path, 'wb') as file:
        pickle.dump(svr, file)


def get_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model


def get_predictions(model, date_from, date_to=None):
    future = pd.date_range(start=date_from, end=date_to, freq='10T')

    df = pd.DataFrame({'Month': future.month, 'DayofWeek': future.dayofweek, "DayofMonth": future.day, "Hour": future.hour, "Minute": future.minute}, index=future)
    X = df.values.tolist()

    predictions = model.predict(X)

    df['Predictions'] = predictions
    df['Predictions'].plot()
    plt.show()

# create_model("..\\Energy_Advice_and_Consumption_Prediction_Dataset.csv", "fullmodel_noYear_noWeek.pickle")
model = get_model("fullmodel_noYear_noWeek.pickle")
get_predictions(model, "19/03/2021", "27/03/2021")


