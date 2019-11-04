import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def DayAheadSARIMAX(endog, exog, date,
                    hoursAhead=38,
                    trend=(0,0,0),
                    seasonal=(1,0,0,24),
                    maxiter=200):
    """
    endog(pd.Series): Series (usually corresponding to a cluster) of hourly indexed load data
    exog(pd.DataFrame): DataFrame of covariates if applicable
    date(datetime-like): Hour from which to begin forecast
    trend(tuple): trend parameters for SARIMAX (p,d,q)
    seasonal(tuple): seasonal parameters for SARIMAX (P,D,Q,m)
    """
    y = endog.copy(deep=True)
    X = exog.copy(deep=True)
    testDate = pd.to_datetime(date)

    y_train = y.loc[y.index < testDate]
    X_train = X.loc[X.index < testDate]
    X_test = X.loc[(testDate <= X.index) & (X.index < testDate+pd.Timedelta(hours=hoursAhead))]

    model = SARIMAX(endog=y_train.values,
                    exog=X_train.values,
                    order=trend,
                    seasonal_order=seasonal,
                    enforce_stationarity=False)
    modelFit = model.fit(disp=0,maxiter=maxiter)
    y_pred = modelFit.forecast(steps=hoursAhead, exog=X_test)
    return y_pred