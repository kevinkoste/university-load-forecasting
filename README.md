# Load forecasting at the university-scale

### Background & Motivation

Yale currently purchases electricity through a procurement arrangement with an independent energy company. This low-risk strategy helps ensure reliability, but costs much more than purchasing electricity directly from wholesale markets.

In addition to administrative barriers, one of the primary challenges of wholesale procurement is accurately forecasting energy demands.

It is very difficult to forecast hour-by-hour electricity and thermal loads of an individual building, because occupants often act unpredictably. At the other end of the spectrum, utility companies use weather and long-term trends to forecast the aggregate load of millions of customers.

This project analyzed Yale’s portfolio of ~250 buildings, an intermediate scale comparable to microgrids, other universities, and small customer choice aggregators across the country.

I built statistical and learning-based forecasting models that use historical data to predict day-ahead electricity demand at hourly intervals. The models are strengthened by the inclusion of powerful date and weather features.

I collaborated with Dr. Clayton Miller, director of the Building and Urban Data Science Lab at the National University of Singapore, to apply the modeling methodology to data from five other universities across the US.


### Highlights

**Working with IoT Data:** IoT devices are only as good as their internet connection. At any given time, approximately 5% of energy meters were not functioning properly. After characterizing the numerous types of errors present in the data, I designed and implemented strategies to remove erronous values and (when appropriate) impute missing values.

**Clustering:** One of the novel conclusions of this project was that separately modeling groups of buildings clustered by daily load-shape can be more accurate than modeling all buildings in aggregate. With the increasing granularity of data offered by smart meters and IoT devices, this strategy will become more effective.

**Machine Learning:** In an effort to assess to effectiveness of learning-based models for multistep time series forecasting, I designed and optimized a multilayer perceptron which performed slightly better than the traditional statistical model (seasonal autoregressive).


### Contents
This repository is a collection of exploratory and experimental analytics notebooks, including scripts for meter error identification, imputation of missing values, load-shape clustering, weather & date feature engineering, and the evaluation of several forecasting strategies.

**Preprocessing:** notebooks for error identification and imputation of electricity, steam, chilled water, and weather data

**Data:** raw and processed electricity, steam, chilled water, and weather data

**Exploration:** notebook for the clustering analysis and a .py module containing useful functions

**Prediction:** notebooks implementing and evaluating various predictive models

