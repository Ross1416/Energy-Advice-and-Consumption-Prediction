from sklearn.ensemble import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct
from sklearn.linear_model import *
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import *

df = pd.read_csv("Unbroken_dataset01.csv")

df.columns = ["Datetime", "Total_Feeder"]

df['Datetime'] = pd.to_datetime(df['Datetime'],
                                format='%Y-%m-%d %H:%M:%S')  # changes the Date_time column to a datetime data type

df.set_index('Datetime', inplace=True)


# df["DayOfYear"] = df.index.dayofyear
df["Hour"] = df.index.hour
df["Minute"] = df.index.minute

df = df.loc["2015-03-13"]


X = df.drop(['Total_Feeder'], axis=1).values
y = df['Total_Feeder'].values

train_split = 0.9
num_train = int(len(X) * 0.9)
X_train = X[:num_train]
X_test = X[num_train:]

y_train = y[:num_train]
y_test = y[num_train:]


reg = RandomForestRegressor(n_estimators = 100, max_depth=5,random_state = 20)
# reg = AdaBoostRegressor(random_state=0, n_estimators=100)
# reg = BaggingRegressor()
# reg = ExtraTreesRegressor(n_estimators=10, random_state=0)
# reg = GradientBoostingRegressor()
# reg = HistGradientBoostingRegressor()
# kernel = DotProduct() + WhiteKernel()
# reg = GaussianProcessRegressor(kernel=kernel, random_state=0)
# reg = LogisticRegression()
# reg = Ridge(alpha=1.0)
# reg = BayesianRidge()
# reg = PoissonRegressor()
# reg = TweedieRegressor()
# reg = GammaRegressor()
# reg = MLPRegressor(random_state=0, max_iter=500)
# reg = DecisionTreeRegressor()
# print(cross_val_score(reg, X, y, cv=10))

reg.fit(X_train,y_train)
accuracy = reg.score(X_test, y_test)
print(accuracy)

predictions = reg.predict(X)

df['Prediction'] = predictions
df['Total_Feeder'].plot()
df['Prediction'].plot()
plt.show()