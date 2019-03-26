# Day-Ahead Predictive Energy Modeling

With accurate predictions of energy demand, Yale could save money by purchasing electricity through competitive bids on the ISO-NE day-ahead market.

Yale currently purchases electricity in real time through a procurement arrangement with the local utility. This low-risk strategy helps ensure reliable and stable electricity, but costs more on average than day-ahead bidding.

Predicting the electricity and heating & cooling loads of an individual building is notoriously difficult. However, a predictive model encompassing Yale’s entire portfolio has the potential to be quite accurate. With more than 250 buildings under management, Yale’s energy demand portfolio is diverse, nonvolatile, and predictable.

The Office of Facilities has been collecting hourly electricity, chilled water, and steam data for almost five years. Until now, the data has mostly been used to identify mechanical system issues, assess energy efficiency initiatives, and bill departments for their energy consumption.

Using machine learning algorithms, the historical data can be leveraged to predict day-ahead electricity demand. The model is strengthened by the inclusion of powerful date and weather predictors

The scope of this project includes meter error identification, imputation of missing values, clustering, weather & date feature engineering, and evaluation of several predictive models.

The work is organized into several Jupyter Notebooks that make use of standard Python data science libraries. Here is a description of the files in each parent folder.

Preprocessing: (3) notebooks for error identification and imputation of electricity, steam, chilled water, and weather data
Data: (8) csv files for raw and processed electricity, steam, chilled water, and weather data
Exploration: (1) notebook for the clustering analysis and (1) py module containing useful functions used throughout the project
Prediction: (4) notebooks implementing and evaluating various machine learning algorithms
