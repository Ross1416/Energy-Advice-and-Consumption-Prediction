# PROOF OF CONCEPT

from Main import *

print("##########################################")
print("##### Energy Consumption Prediction ######")
print("##########################################")
print("")
print("Please enter a date range to receive a prediction: ")
print("Date must be in the format dd/mm/yyyy")

print("Start date")
start_date = input("> ")
print("End date")
end_date = input("> ")

model = GetModel("Models/fullmodel_noYear_noWeek.pickle")
df = MakePredictions(model, start_date, end_date)

PlotPredictions(df)



