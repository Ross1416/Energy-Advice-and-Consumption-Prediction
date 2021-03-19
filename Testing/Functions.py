import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

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

model = get_model("fullmodel_noYear_noWeek.pickle")
get_predictions(model, "19/03/2021", "27/03/2021")


