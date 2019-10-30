"""
Reads raw metadata and load data from Building Data Genome Project files bdgp-meta.csv and bdgp-loads.csv
Writes new .csv for each campus of length 8760, filling NA (very few) with basic bfill method
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ingest and format metadata for four big campuses
metaRaw = pd.read_csv('../../data/raw/bdgp/bdgp-meta.csv',index_col=0)
meta = pd.DataFrame(index=['arizona','michigan','newYork','newJersey'],
                    columns=['weatherfile'],
                    data=['weather0.csv','weather3.csv','weather2.csv','weather4.csv'])

meta['buildings'] = [list(metaRaw[metaRaw['newweatherfilename'] == meta.loc[k]['weatherfile']].index) for k in meta.index]
meta['start'] = [pd.to_datetime(metaRaw[metaRaw['newweatherfilename'] == meta.loc[k]['weatherfile']]['datastart'].iloc[0],format='%d/%m/%y %H:00') for k in meta.index]
meta['end'] = [pd.to_datetime(metaRaw[metaRaw['newweatherfilename'] == meta.loc[k]['weatherfile']]['dataend'].iloc[0],format='%d/%m/%y %H:00') for k in meta.index]


# ingest and format load data
data = pd.read_csv('../../data/raw/bdgp/bdgp-elec.csv',index_col=0)
data.index = pd.to_datetime(data.index,format='%Y-%m-%d %H:00:00+00:00')

# keeping these weather columns
weatherDict = {'temp':'TemperatureC', 'rh':'Humidity', 'dewpoint':'Dew PointC', 'precip':'Precipitationmm',}

# write new csvs
for campus in meta.index:
    
    # load data
    load = data[meta.loc[campus]['buildings']].loc[meta.loc[campus]['start']:meta.loc[campus]['end']]
    load.fillna(method='bfill',inplace=True)
    del load.index.name
    load.to_csv(f'../../data/processed/{campus}-loads.csv')

    # ingesting weather data
    weather = pd.read_csv('../../data/raw/bdgp/weather/'+meta.loc[campus]['weatherfile'],index_col=0,date_parser=pd.to_datetime)
    weather.index = weather.index.floor(freq='H')
    weather = weather.loc[~weather.index.duplicated(keep='first')]
    for k in weather.columns:
        if weather[k].dtype == object: continue
        weather[k] = weather[k][np.abs(weather[k]-weather[k].mean()) < 10*weather[k].std()]
    weather = weather.reindex(load.index, method='pad').fillna(method='bfill').fillna(method='pad')
    
    # refactoring to covariate matrix (aka big X)
    covariates = pd.DataFrame(index=weather.index)
    for key in weatherDict:
        covariates[key] = weather[weatherDict[key]].values

    # add calendar fields
    for i in range(0,24):
        covariates['hour_'+str(i)] = (covariates.index.hour.values == i).astype(int)
    for i in range(0,7):
        covariates['day_'+str(i)] = (covariates.index.dayofweek.values == i).astype(int)

    # scale each feature from 0 to 1
    scaler = MinMaxScaler()
    covariates = pd.DataFrame(scaler.fit_transform(covariates.values),columns=covariates.columns,index=covariates.index)
    
    covariates.to_csv(f'../../data/processed/{campus}-covariates.csv')
