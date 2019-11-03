import numpy as np
import pandas as pd
import datetime as dt

from src.processing.clustering import LoadShapeCluster
from src.training.sarimax import TrainSARIMAX

## hyperparameters
location = 'arizona'
numClusters = 1
trendParams = (0,0,0)
seasonalParams = (1,0,0,24)
maxiter = 50

# import processed load and covariate data
loads = pd.read_csv('data/processed/'+location+'-loads.csv',index_col=0,date_parser=pd.to_datetime)
covariates = pd.read_csv('data/processed/'+location+'-covariates.csv',index_col=0,date_parser=pd.to_datetime)

# robust method to get today's date in the correct year (based on df.index)
today = dt.datetime.now().date()
testYear = loads.index[(loads.index.month == today.month) & (loads.index.day == today.day)].year[0]
testDate = pd.to_datetime(dt.date(year=testYear, month=today.month, day=today.day))+pd.Timedelta(hours=10)

# apply clustering on previous week of data
clusterTestDf = loads.loc[(testDate-pd.Timedelta(days=7) <= loads.index) & (loads.index < testDate)]
clusterMap, clusterScore = LoadShapeCluster(clusterTestDf,numClusters)

# define test df based on clustering results
df = pd.DataFrame(data=loads.sum(axis=1), index=loads.index, columns=['aggregate'])
for i in range(1,numClusters+1):
    df['cluster'+str(i)] = loads[[k for k in loads.columns if clusterMap[k]==i]].sum(axis=1)

# generate MultiIndex to store results, initialize results df
indexMulti = pd.MultiIndex.from_product([list(df.columns),['actual','sarimax','mlp']])
results = pd.DataFrame(index=pd.date_range(start=testDate, freq='H', periods=38), columns=indexMulti)

# train a model, make a forecast, and store the results for each cluster
for cluster in df.columns:
    # define and split data
    y = df[cluster].copy(deep=True)
    X = covariates.copy(deep=True)
    y_train = y.loc[y.index < testDate]
    y_test = y.loc[(testDate <= y.index) & (y.index < testDate+pd.Timedelta(hours=38))]
    X_train = X.loc[X.index < testDate]
    X_test = X.loc[(testDate <= X.index) & (X.index < testDate+pd.Timedelta(hours=38))]

    # train the SARIMAX model for this cluster, make a forecast
    modelFit = TrainSARIMAX(endog=y_train, exog=X_train, trend=trendParams, seasonal=seasonalParams, maxiter=maxiter)
    forecast = modelFit.forecast(steps=38, exog=X_test)
    
    # store the results
    results[cluster,'actual'] = y_test.values
    results[cluster,'sarimax'] = forecast

results.to_csv('data/results/'+str(today)+'.csv')