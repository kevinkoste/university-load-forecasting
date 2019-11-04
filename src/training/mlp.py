import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def DayAheadMLP(endog, exog, date,
                lags=range(1,169),
                hoursAhead=38,
                epochs=200,
                activation='relu',
                optimizer='adam',
                loss='mse',
                verbose=0):
    """
    Trains a fresh MLP and returns a day-ahead forecast 
    endog(pd.Series): Series (usually corresponding to a cluster) of hourly indexed load data
    exog(pd.DataFrame): DataFrame of covariates if applicable
    date(date-like obj): Hour after which to begin 38-hour-ahead forecast

    lags(List): List of lag features to generate (default=range(1,169))
    hoursAhead(int): number of hours ahead to predict (default=38)
    epochs(int): number of epochs for training (default=200)
    activation(str): key for activation function (default='relu')
    optimizer(str): key for optimizer (default='adam')
    loss(str): key for loss function (default='mse')
    """
    # force DataFrame dtype for y, copy X
    y = pd.DataFrame(endog)
    X = exog.copy(deep=True)
    testDate = pd.to_datetime(date)

    # scale y (0,1) on annual min and max, important assumption
    scaler = MinMaxScaler().fit(y)
    y = pd.DataFrame(data=scaler.transform(y), index=y.index, columns=y.columns)

    for i in lags:
        X['t-'+str(i)] = y.iloc[:,0].shift(i)
    for j in range(1,38):
        y['t+'+str(j)] = y.iloc[:,0].shift(-j)

    # truncate on both sides to remove NaNs
    X.dropna(inplace=True)
    y = y.reindex(X.index, axis=0).dropna()
    X = X.reindex(y.index, axis=0).dropna()

    # train/test split, train range includes all available data up to two days prior to test date
    y_train = y.loc[y.index < testDate-pd.Timedelta(days=2)].values
    X_train = X.loc[X.index < testDate-pd.Timedelta(days=2)].values
    X_test = X.loc[X.index == testDate].values

    # set input and hidden layer dimensions
    inputDim = X_train.shape[1]
    hiddenDim = (inputDim-38)//2 + 38

    # define model based on hyperparameters
    model = Sequential()
    model.add(Dense(hiddenDim, activation=activation, input_dim=inputDim))
    model.add(Dense(38))
    model.compile(optimizer=optimizer, loss=loss)

    # fit the model and make a prediction
    model.fit(X_train, y_train, epochs=epochs)
    y_pred = model.predict(X_test, verbose=0)

    # return result after reverse scaling
    return scaler.inverse_transform(y_pred).flatten()
