import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

df = pd.read_csv("Unbroken_dataset01.csv", )

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)