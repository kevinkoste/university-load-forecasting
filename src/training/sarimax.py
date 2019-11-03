import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def TrainSARIMAX(endog, exog, trend, seasonal, maxiter=200):
    """
    endog(pd.Series): Series (usually corresponding to a cluster) of hourly indexed load data
    exog(pd.DataFrame): DataFrame of covariates if applicable
    trend(tuple): trend parameters for SARIMAX (p,d,q)
    seasonal(tuple): seasonal parameters for SARIMAX (P,D,Q,m)
    """
    y_train = endog.copy(deep=True)
    X_train = exog.copy(deep=True)

    model = SARIMAX(endog=y_train.values,
                    exog=X_train.values,
                    order=trend,
                    seasonal_order=seasonal,
                    enforce_stationarity=False)
    model_fit = model.fit(disp=0,maxiter=maxiter)
    return model_fit