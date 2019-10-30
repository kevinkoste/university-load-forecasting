import numpy as np
import pandas as pd
import datetime as dt

from src.processing.clustering import LoadShapeCluster
from src.training.sarimax import TrainSARIMAX

# This script is designed to execute ONE (for now) day-ahead forecast given a set of parameters
# It outputs the result in the /temp folder

# hyperparameters
location = 'arizona'
numClusters = 1
trendParams = (0,0,0)
seasonalParams = (1,0,0,24)

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

results = pd.DataFrame(index=pd.date_range(start=testDate, freq='H', periods=38))

for cluster in df.columns:
    
    y = df[cluster].copy(deep=True)
    X = covariates.copy(deep=True)

    # split data
    y_train = y.loc[y.index < testDate]
    y_test = y.loc[(testDate <= y.index) & (y.index < testDate+pd.Timedelta(hours=38))]
    X_train = X.loc[X.index < testDate]
    X_test = X.loc[(testDate <= X.index) & (X.index < testDate+pd.Timedelta(hours=38))]

    # train the SARIMAX model for this cluster, then make a forecast
    modelFit = TrainSARIMAX(endog=y_train, exog=X_train, trend=trendParams, seasonal=seasonalParams)
    forecast = modelFit.forecast(steps=38, exog=X_test)
    
    # store the results
    results[cluster+'_actual'] = y_test.values
    results[cluster+'_sarimax'] = forecast

results.to_csv('data/results/'+str(today)+'.csv')