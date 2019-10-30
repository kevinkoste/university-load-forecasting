# currently broken, Yale data is very low quality

import pandas as pd
import numpy as np

import utility_functions as fn

# manually define building names
buildingnames = ['YUAG',
                 'Berkeley',
                 'Hopper',
                 '304Elm',
                 'Davenport',
                 '38HH',
                 '320Temple',
                 '53Wall',
                 'Sprague',
                 'Malone',
                 'Trumbull',
                 '17HH']


# read elec data from csv into a new dataframe
raw = pd.read_csv('../data/elec_raw.csv',index_col=0,na_values=['#########'])

# remove precalculated demand values, which tend to be bugged
raw = raw.drop(raw.columns[np.arange(0,len(buildingnames)*2,2)], axis=1)

# reindex appropriately by hourly datetime
raw.index = pd.to_datetime(raw.index,format='%a %m/%d/%y %H:00')

# add missing rows by full reindexing
correct_dt = pd.DatetimeIndex(start='2018-01-01 00:00:00',end='2018-07-27 23:00:00',freq='h')
raw = raw.reindex(index=correct_dt)

# rename columns
raw.columns = buildingnames


# create "raw" demand dataframe for plotting later on
raw_plots = raw.diff().drop(raw.index[0])


# remove impossible outliers based on negative percent change
raw_head = raw.iloc[0]
raw = raw.where(raw.pct_change(limit=1)>0)
raw.iloc[0] = raw_head



# interpolate gaps in consumption data 6 hours and shorter, optional
for k in raw.columns:
    raw[k] = fn.limited_impute(raw[k],6)


# ## Error Identification & Removal


# create a new dataframe for hourly demand (kW)
demand = raw.diff().drop(raw.index[0])

# save head to replace later
demand_head = demand.iloc[0:4]

errors = demand.isnull().sum()
print('Missing values:')
print(errors)

# fn.plot_all(demand,'2018-01-01 01:00:00','2018-07-27 23:00:00')


# remove huge statistical outliers
# demand = demand.where(demand > demand.median() - 2.5*demand.std())
# demand = demand.where(demand < demand.median() + 5*demand.std())

new_errors = demand.isnull().sum() - errors
print('New missing values from this step:')
print(new_errors)
errors = demand.isnull().sum()

fn.plot_all(demand,'2018-01-01 01:00:00','2018-07-27 23:00:00')


# remove errors by rolling min and max in 15-day chunks
chunk_size = 360
i=0

while i < len(demand):
    end = i+chunk_size
    if end > len(demand): end = len(demand)
    demand[i:end].where(demand[i:end] > demand[i:end].rolling(18).min().median()*0.7, inplace=True)
    demand[i:end].where(demand[i:end] < demand[i:end].rolling(18).max().median()*1.3, inplace=True)
    i = i+chunk_size


new_errors = demand.isnull().sum() - errors
print('New missing values from this step:')
print(new_errors)
errors = demand.isnull().sum()

fn.plot_all(demand,'2018-01-01 01:00:00','2018-07-27 23:00:00')


# ## Imputation of missing values


dense = demand.copy(deep=True)

# interpolate gaps shorter than 6 hours
for k in dense.columns:
    dense[k] = fn.limited_impute(dense[k],6)

print('Number of missing values remaining:')
print(dense.isnull().sum())


# drop 320Temple for now, data in July is too messy for sinusoidal interpolation
dense = dense.drop('320Temple', axis=1)

# interpolate gaps longer than 6 hours using least-squares optimized sinusoidal fit
for k in dense.columns:
    dense[k] = fn.sine_impute(dense[k])

# ## Export

# replace head for final export
dense.iloc[0:4] = demand_head.drop('320Temple',axis=1)

# export clean data to csv
# dense.round(1).to_csv('../data/elec_clean.csv')

