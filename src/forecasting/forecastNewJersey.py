import numpy as np
import pandas as pd
import datetime as dt

from src.processing.clustering import LoadShapeCluster
from src.training.sarimax import DayAheadSARIMAX
from src.training.mlp import DayAheadMLP

# This script executes SARIMAX and MLP day-ahead forecasts for the current day,
# and stores the results in the results folder

# general parameters
location = 'newJersey'
numClusters = 3

# SARIMA hyperparameters
trendParams = (0,0,0)
seasonalParams = (1,0,0,24)
maxiter = 10

# MLP hyperparameters
lags = range(1,169)
epochs = 10
activation = 'relu'
optimizer='adam'
loss='mse'

# import preprocessed load and covariate data
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
clusterNames = list(df.columns)
indexMulti = pd.MultiIndex.from_product([clusterNames,['actual','sarimax','mlp']])
results = pd.DataFrame(index=pd.date_range(start=testDate, freq='H', periods=38), columns=indexMulti)

for cluster in clusterNames:
    # grab data for a single cluster, save the test data
    y = df[cluster].copy(deep=True)
    X = covariates.copy(deep=True)
    results[cluster,'actual'] = y.loc[(testDate<=y.index) & (y.index<testDate+pd.Timedelta(hours=38))]

    # use DayAheadSARIMAX helper function to train a fresh model and make a day-ahead forecast
    y_pred_sarimax = DayAheadSARIMAX(
        endog=y,
        exog=X,
        date=testDate,
        trend=trendParams,
        seasonal=seasonalParams,
        maxiter=maxiter)
    results[cluster,'sarimax'] = y_pred_sarimax
    
    # use DayAheadMLP helper function to train a fresh model and make a day-ahead forecast
    y_pred_mlp = DayAheadMLP(
        endog=y,
        exog=X,
        date=testDate,
        lags=lags,
        epochs=epochs,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
        verbose=0)
    results[cluster,'mlp'] = y_pred_mlp

# calculate the cluster sum results here
clusterNames.remove('aggregate')
results['clustersum','actual'] = results[clusterNames].swaplevel(axis=1)['actual'].sum(axis=1)
results['clustersum','sarimax'] = results[clusterNames].swaplevel(axis=1)['sarimax'].sum(axis=1)
results['clustersum','mlp'] = results[clusterNames].swaplevel(axis=1)['mlp'].sum(axis=1)
results.to_csv('data/forecasts/'+location+'/'+str(today)+'.csv')