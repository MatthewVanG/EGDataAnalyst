import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

## Formats the data in the provided CSV data frame from string form into a numerical form
## Additionally it will replace any NaN values with the average value for that feature.
## Some features may be more accurate to take the median as a step for future work.
def formatdata(df, features):
  for feature in features:
    df[feature]= pd.to_numeric(df[feature], errors="coerce")
    average = df[feature][df[feature].notna()].mean()
    df[feature] = df[feature].fillna(average)
  return df

## Performs a series of ridge regression given a series of penalty values
def ridgeregression(lambdas, train_data, validate_data, features):
  data = []
  for lmdb in lambdas:
    ridge_model = Ridge(alpha=lmdb)
    ridge_model_fit = ridge_model.fit(train_data[features],train_data["LeagueIndex"])
    predict_train = ridge_model_fit.predict(train_data[features])
    train_rmse = np.sqrt(mean_squared_error(train_data["LeagueIndex"], predict_train))
    predict_validate = ridge_model_fit.predict(validate_data[features])
    validate_rmse = np.sqrt(mean_squared_error(validate_data["LeagueIndex"], predict_validate))
    data.append({
        'l2_penalty' : lmdb,
        'model' : ridge_model_fit,
        'train_rmse' : train_rmse,
        'validation_rmse' : validate_rmse
    })
  ridge_data = pd.DataFrame(data)
  return ridge_data

## Performs a series of LASSO regression given a series of penalty values
def lassoregression(lambdas, train_data, validate_data, features):
  data = []
  for lmdb in lambdas:
    LASSO_model = Lasso(alpha=lmdb)
    LASSO_model_fit = LASSO_model.fit(train_data[features],train_data["LeagueIndex"])
    predict_train = LASSO_model_fit.predict(train_data[features])
    train_rmse = np.sqrt(mean_squared_error(train_data["LeagueIndex"], predict_train))
    predict_validate = LASSO_model_fit.predict(validate_data[features])
    validate_rmse = np.sqrt(mean_squared_error(validate_data["LeagueIndex"], predict_validate))
    data.append({
        'l1_penalty' : lmdb,
        'model' : LASSO_model_fit,
        'train_rmse' : train_rmse,
        'validation_rmse' : validate_rmse
    })
  lasso_data = pd.DataFrame(data)
  return lasso_data