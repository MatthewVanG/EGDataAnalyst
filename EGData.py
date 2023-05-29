import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from functions import formatdata, ridgeregression, lassoregression

"""Youre given a dataset of Starcraft player performance data in ranked games. We want
to develop a model to predict a players rank using the information provided in the
dataset."""

## Because there is no player ID asisgned to the data it would be best to evaluate each game on its
## own and see how that performance in the singular game relates to the rank of the player.
df = pd.read_csv('starcraft_player_data.csv')

## Use data imputation to figure out what values to put in for the ?. Since all values are numerical
## the simplest approach is to use the average of each column
## need to convert strings to ints
df = df.replace("?", np.NaN)
df = formatdata(df, df.columns.tolist())


## Perform a basic linear regression test to determine the baseline comparisson for the ridge
## regression and LASSO regression models.
train_data, test_data_and_validate_data = train_test_split(df, test_size=0.2)
test_data, validate_data = train_test_split(test_data_and_validate_data, test_size=0.5)
basic_features = ["Age", "HoursPerWeek", "TotalHours", "APM", "SelectByHotkeys", "AssignToHotkeys", "UniqueHotkeys", "MinimapAttacks", "MinimapRightClicks", "NumberOfPACs", "GapBetweenPACs", "ActionLatency", "ActionsInPAC", "TotalMapExplored", "WorkersMade", "UniqueUnitsMade", "ComplexUnitsMade", "ComplexAbilitiesUsed"]
basic_model = LinearRegression().fit(train_data[basic_features], train_data["LeagueIndex"])
basic_predict_train = basic_model.predict(train_data[basic_features])
train_rmse_basic = np.sqrt(mean_squared_error(train_data["LeagueIndex"], basic_predict_train))
basic_predict_test = basic_model.predict(test_data[basic_features])
test_rmse_basic = np.sqrt(mean_squared_error(test_data["LeagueIndex"], basic_predict_test))

## Be able to read out what the RMSE are for the training dataset and test dataset
# print(train_rmse_basic)
# print(test_rmse_basic)

## Produce a table version of the data to see how the predication can vary from the actual rank
# output = pd.DataFrame(np.array([test_data["LeagueIndex"], basic_predict_test, test_data["LeagueIndex"] - basic_predict_test]))
# output = output.transpose()
# output.columns = ["Actual Rank", "Predicted Rank", "Difference"]
# print(output)

## Standardize the data for use in the Ridge and Lasso Regression Methods
scaler = preprocessing.StandardScaler().fit(train_data[basic_features])
train_data[basic_features] = scaler.transform(train_data[basic_features])
validate_data[basic_features] = scaler.transform(validate_data[basic_features])
test_data[basic_features] = scaler.transform(test_data[basic_features])

## Ridge Regression Test
## Define a range of penlty values to find which one is the best
l2_lambdas = np.logspace(-10, 10, 30, base = 10)
ridge_data = ridgeregression(l2_lambdas, train_data, validate_data, basic_features)

## Plot the results of the regression to see the differences in penalty values.
plt.plot(ridge_data['l2_penalty'], ridge_data['validation_rmse'],
         'b-^', label='Validation')
plt.plot(ridge_data['l2_penalty'], ridge_data['train_rmse'],
         'r-o', label='Train')
plt.xscale('log')
plt.xlabel('l2_penalty')
plt.ylabel('RMSE')
plt.legend()
plt.show()


## Lasso Regression Test
## Define a range of penlty values to find which one is the best
l1_lambdas = np.logspace(-10, 10, 30, base = 10)
lasso_data = lassoregression(l1_lambdas, train_data, validate_data, basic_features)

## Plot the results of the regression to see the differences in penalty values.
plt.plot(lasso_data['l1_penalty'], lasso_data['validation_rmse'],
         'b-^', label='Validation')
plt.plot(lasso_data['l1_penalty'], lasso_data['train_rmse'],
         'r-o', label='Train')
plt.xscale('log')
plt.xlabel('l1_penalty')
plt.ylabel('RMSE')
plt.legend()
plt.show()

## Since the LASSO regression method can completely eliminate features we want to investigate its
## results more closely. After doing the LASSO regression we want to find the best penalty value
## then using that best model make some calculations against the test dataset.
index = lasso_data['validation_rmse'].idxmin()
row = lasso_data.loc[index]
best_l1 = row['l1_penalty']
model = row['model']
predict_test = model.predict(test_data[basic_features])
test_rmse_lasso = np.sqrt(mean_squared_error(test_data["LeagueIndex"], predict_test))

## Print the penatly value, RMSE, and coefficients for the given model
print(best_l1)
print(test_rmse_lasso)
print(model.coef_)