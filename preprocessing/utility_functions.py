import pandas as pd
import numpy as np
from scipy import optimize

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import utility_functions as fn


def sine_fit(tt, yy):
    """
    This is adapted from the fit_sin function from stackoverflow

    Fits sinusoid to the input time sequence
    Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fit_func = lambda t: A * np.sin(w*t + p) + c
    return {'fit':fit_func,'amp':A,'omega':w,'phase':p,'offset':c,'freq':f}


def sine_impute(series_in):
    """
    Imputes all gaps using a least-squares optimized sinusoidal fit function
    
    series_in: Series or column of DataFrame to to be imputed
    """
    import utility_functions as fn
    series = series_in.copy(deep=True)
    
    gaps = fn.gaps(series)
    
    # iterate through list of gaps and fill each
    for i in gaps.index:
        fit_data = series[gaps.start_int[i] - 3*gaps.length[i]:gaps.start_int[i]]
        fit = fn.sine_fit(np.arange(len(fit_data)),fit_data.values)

        imputed_values = fit['fit'](np.arange(len(fit_data)+1,len(fit_data)+gaps.length[i]+1))
        series[gaps.start_int[i]:gaps.end_int[i]] = imputed_values
    
    return series


def add_hours_before(df_in,hours_before):
    """
    Adds new features to a DataFrame with data from some number of hours before the target hour
    
    df_in: dataframe with hourly indexed values
    hours_before: array of the previous hours to consider; e.g. [16,17,18] or np.arange(1,25)
    returns: df with appropriate columns added
    """
    df = df_in.copy(deep=True)
    for i in hours_before:
        for j in range(len(df_in.columns)):
            df[df.columns[j]+'-'+str(i)] = np.append(np.array([np.nan]*i), df[df.columns[j]].values[i-1:len(df)-1])
    return df


def date_features(df_in, label=None):
    """
    Creates time series features from datetime index
    """
    df = df_in.copy(deep=True)
    df = df.drop(df.columns,axis=1)
    
    df['date'] = df.index
    df['hourofday'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hourofday','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


def plot_feature(df, column, start, stop, **keyword_parameters):
    """
    Plots all features for a snapshot between two indices
    
    df: DataFrame with datetime[ns] index
    column: str with column name
    start: str of index label at which to start plot
    stop: str of index label at which to end plot
    """
    if ('ylabel' in keyword_parameters):
        ylabel = keyword_parameters['ylabel']
    else:
        ylabel = ''
        
    start_int = df.index.get_loc(start)
    stop_int = df.index.get_loc(stop)

    snapshot = df.iloc[start_int:stop_int]

    plot = plt.plot(snapshot.index, snapshot[column],'ro')

    plt.ylabel(ylabel)
    plt.xlabel('')
    plt.xticks(rotation=20)
    plt.setp(plot, markersize=5)
    plt.show()
    return
    
    
def plot_all(df, start, stop, **keyword_parameters):
    """
    Plots all features for a snapshot between two indices
    
    df: DataFrame with datetime[ns] index
    start: str of index label at which to start plot
    stop: str of index label at which to end plot
    """
    if ('style' in keyword_parameters):
        style = keyword_parameters['style']
    else:
        style = 'r-'
        
    if ('ylabel' in keyword_parameters):
        ylabel = keyword_parameters['ylabel']
    else:
        ylabel = ''
        
    start_int = df.index.get_loc(start)
    stop_int = df.index.get_loc(stop)

    snapshot = df.iloc[start_int:stop_int+1]

    fig = plt.figure(figsize=(20, 5*len(snapshot.columns)),dpi= 80,facecolor='w')

    for i in range(1,len(snapshot.columns)+1):
        ax = fig.add_subplot(len(snapshot.columns),3,i)
        ax.title.set_text(snapshot.columns[i-1])
        ax.set_ylabel(str(ylabel))
        ax.plot(snapshot[snapshot.columns[i-1]],style)
    return


def gaps(series,**keyword_parameters):
    """
    Returns DataFrame indicating start, end, and length of gaps longer than limit
    
    series: Series/column to be checked, with datetime[ns] index
    gap_length: int maximum allowable gap length in hours
    """
    if ('limit' in keyword_parameters):
        limit = keyword_parameters['limit']
    else:
        limit=0
    
    nans = series.notna()
    gaps_list = []
    
    for i in range(len(nans)):
        if (nans[i-1] == True) & (nans[i] == False):
            dict1 = {}
            for j in range(i,len(nans)):
                if nans[j]==True:
                    break
            if j-i > limit:
                dict1.update({'start_int':i, 'end_int':j, 'length':j-i, 'start':nans.index[i],'end':nans.index[j]}) 
                gaps_list.append(dict1)
                
    if gaps_list == []:
        return pd.DataFrame(columns=['start_int','end_int','length','start','end'])
    else:
        return pd.DataFrame(gaps_list)[['start_int','end_int','length','start','end']]

    
def limited_impute(series,limit,**keyword_parameters):
    """
    Returns Series with gaps shorter than the limit interpolated based on the method given
    
    series: Series/column to be checked, with datetime[ns] index
    limit: int maximum gap length to be interpolated, in hours
    """
    if ('method' in keyword_parameters):
        method = keyword_parameters['method']
    else:
        method='linear'

    # this is an implementation of the gaps function within this function, to avoid that dependency
    nans = series.notna()
    gaps_list = []
    
    for i in range(len(nans)):
        if (nans[i-1] == True) & (nans[i] == False):
            dict1 = {}
            for j in range(i,len(nans)):
                if nans[j]==True:
                    break
            if j-i > limit:
                dict1.update({'start_int':i, 'end_int':j, 'length':j-i, 'start':nans.index[i],'end':nans.index[j]}) 
                gaps_list.append(dict1)
     
    if gaps_list == []:
        gaps =  pd.DataFrame(columns=['start_int','end_int','length','start','end'])
    else:
        gaps =  pd.DataFrame(gaps_list)[['start_int','end_int','length','start','end']]
        
    # this is the start of the interpolation section
    df = pd.DataFrame(series)
    df['range'] = np.arange(len(df))

    exclude = np.array([])

    for i in range(len(gaps)):
        temp = np.arange(gaps['start_int'][i],gaps['end_int'][i])
        exclude = np.append(exclude,temp).astype(int)

    df_out = (df.mask(df['range'].isin(exclude),df.interpolate(method=method,inplace=True))
                .astype(float)
                .drop(labels='range',axis='columns'))

    return df_out[df.columns[0]]


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculates MAPE given y_true and y_pred
    
    y_true: array-like actual values
    y_pred: array-like predicted values
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -------------------------------------------------------------------------------------------
# The following functions are not currently being used in any pipelines, but may be revisited
# -------------------------------------------------------------------------------------------

# def squared_sine_fit(tt, yy):
#     """
#     This is adapted from the fit_sin function
#     Trying to add a squareness factor - this matches energy data more closely than pure sinusoid

#     Fits sinusoid to the input time sequence
#     Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
#     """
#     tt = np.array(tt)
#     yy = np.array(yy)
#     ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(np.fft.fft(yy))
#     guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = np.std(yy) * 2.**0.5
#     guess_offset = np.mean(yy)
#     guess_squareness = 1
#     guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_squareness, guess_offset])

#     def sinfunc(t, A, w, p, m, c):  return A * np.sin(w*t + p)**m + c
#     popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, m, c = popt
#     f = w/(2.*np.pi)
#     fit_func = lambda t: A * np.sin(w*t + p)**m + c
#     return {'fit':fit_func,'amp':A,'omega':w,'phase':p,'squareness':m,'offset':c,'freq':f}




# def sine_fit(series):
#     """
#     NEEDS TO BE CHANGED to take just a series
#     Fits a sinusoid to evenly-spaced time series data
    
#     df: DataFrame with datetime[ns] index
#     column: str name of column in question
#     start: int index location of first value in the impute range
#     end: int index location of last value in the impute range
#     Return fitting parameter 'fitfunc'
#     """
#     y = series.values
#     t = np.arange(len(series))
    
#     f = np.fft.fftfreq(len(t), (t[1]-t[0]))
#     Fy = abs(np.fft.fft(y))
    
#     guess_freq = abs(f[np.argmax(Fy[1:])+1])
#     guess_amp = np.std(y) * 2.**0.5
#     guess_offset = np.mean(y)
#     guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    
#     def sine(x, A, w, p, c):  return A * np.sin(w*x + p) + c
    
#     popt, pcov = optimize.curve_fit(sine, t, y, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*np.pi)
#     fit_function = lambda x: A * np.sin(w*x + p) + c
    
#     return {'fit':fit_function,'amp':A,'omega':w,'phase':p,'offset':c,'freq':f}



# def sine_impute(df_in,column):
#     """
#     Imputes all gaps using a least-squares optimized sinusoidal fit function
#     Old, works with series not df now 
    
#     df_in: DataFrame to which the target column belongs
#     column: int index of column to be imputed
#     """
#     df_out = df_in.copy(deep=True)
#     gaps = fn.gaps(series)

#     for i in gaps.index:

#         data_start_index = gaps.start[i] - 2*gaps.length[i]-1
#         data_end_index = gaps.start[i]-1
#         impute_start_index = gaps.start[i]
#         impute_end_index = gaps.end[i]
#         # print(gaps.start[i])
#         # print(gaps.end[i])

#         x_start = 0
#         x_end = data_end_index - data_start_index
#         x_impute_start = x_end+1
#         x_impute_end = impute_end_index - data_start_index

#         fit_function = sine_fit(df_in,df_in.columns[column],data_start_index,data_end_index)
        
#         imputed_values = fit_function['fit'](np.arange(x_impute_start,x_impute_end))
#         # print(imputed_values)

#         df_out.iloc[gaps.start[i]:gaps.end[i],[column]] = pd.DataFrame(imputed_values).values
#     return df_out



# def gap_finder_broken(df_in,gap_length):
#     """
#     BROKEN, cannot handle gaps shorter than 1.5*gap+length
#     Returns a bool DataFrame indicating gaps longer than gap_length hours in df_in
    
#     df: DataFrame to be checked, with datetime[ns] index
#     gap_length: int maximum allowable gap length in hours
#     """
#     fward = (df_in.notna()
#                   .rolling(gap_length,min_periods=1)
#                   .sum()
#                   .astype(bool)
#             )
    
    
#     bward = (df_in.iloc[::-1]
#                   .notna()
#                   .rolling(gap_length,min_periods=1)
#                   .sum()
#                   .iloc[::-1]
#                   .astype(bool)
#             )
    
#     return fward & bward


# def gap_finder_wshift(df_in,gap_length):
#     """
#     trying to fix short gap problem using bitwise operators.. failing
#     """
#     df = df_in.notna()
#     return (df | df.shift(-1) | df.shift(-2) | df.shift(-3)) & (df | df.shift(1) | df.shift(2) | df.shift(3))    
    

    
# def column_gaps(df_in,column,max_length):
#     """
#     Returns DataFrame indicating the start, end and length of gaps in one column of a DataFrame
#     This is a handy combination of two other utility functions: gap_finder and gaps_as_df
    
#     df_in: DataFrame to which the target column belongs
#     column: str name of column to be checked
#     max_length: int maximum length of gaps to be ignored
#     """
#     bool_gaps = gap_finder(df_in,column,max_length)
#     return gaps_as_df(bool_gaps,column)



# def gap_finder(df_in,column,max_length):
#     """
#     SLOW LOOP-BASED, could not find a quicker array/arithmetic-based solution
#     Returns a bool DataFrame indicating gaps longer than gap_length hours in column of df_in
    
#     df: DataFrame to be checked, with datetime[ns] index
#     column: str name of column to be modified
#     max_length: int maximum allowable gap length in hours
#     """
#     df_out = df_in.copy(deep=True)
#     df_out[:] = True

#     df = df_in.notna()
    
#     for i in range(len(df)):
#             if (df[column][i-1]==True) & (df[column][i]==False):
#                 gap_start=i
#                 for j in range(i,len(df)):
#                     if df[column][j]==True:
#                         gap_end=j
#                         break
#                 if gap_end-gap_start > max_length:
#                     df_out[column][i:j] = False

#     return df_out


# def gaps_as_df(gap_indicator,column):
#     """
#     Returns DataFrame indicating the start, end and length of gaps in gap_indicator
    
#     gap_indicator: DataFrame to be checked, with datetime[ns] index
#     column: int maximum allowable gap length in hours
#     """
#     gaps_list = []
    
#     for i in range(len(gap_indicator[column])):
#         if (gap_indicator[column][i-1]==True) & (gap_indicator[column][i]==False):
#             dict1 = {}
#             for j in range(i,len(gap_indicator[column])):
#                 if gap_indicator[column][j]==True:
#                     break
#             dict1.update({'start': i,'end': j,'length': j-i}) 
#             gaps_list.append(dict1)

#     return pd.DataFrame(gaps_list)[['start','end','length']]
    
    
    
# def print_gaps(df,gap_indicator_column):
#     """
#     Prints integer indices of starts and ends of all gaps indicated by the gap_indicator_column
    
#     df: DataFrame with datetime[ns] index
#     gap_indicator_column: str giving name of the column that indicates gaps using True or False
#     """
    
#     print('start | end')
    
#     for i in range(len(df)):
#         if (df[gap_indicator_column][i-1]==True) & (df[gap_indicator_column][i]==False):
#             gap_start=i
#             for j in range(i,len(df)):
#                 if df[gap_indicator_column][j]==True:
#                     gap_end=j
#                     break
#             print(gap_start, gap_end)


# def plotweather(df, start, stop):
#     """
#     Plots all features for a snapshot between two indices
#     df: DataFrame with datetime[ns] index
#     start: str of index label at which to start plot
#     stop: str of index label at which to end plot
#     """
    
#     start_int = df.index.get_loc(start)
#     stop_int = df.index.get_loc(stop)

#     snapshot = df.iloc[start_int:stop_int]
    
#     fig, (p1,p2,p3,p4,p5,p6,p7,p8,p9) = plt.subplots(nrows=3,ncols=3,sharex=True)
        
#     p1.scatter(x=snapshot.index,y=snapshot['temp'])
#     p1.set_title('temp')
#     p1.set_yaxis('Temperature [K]')
    
#     p2.scatter(x=snapshot.index,y=snapshot['t_min'])
#     p2.set_title('t_max')
#     p2.set_yaxis('Temperature [K]')
    
    
# def fit_sin(tt, yy):
#     """
#     This is the original fit_sin function from stackoverflow
#
#     Fits sinusoid to the input time sequence
#     Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
#     """
#     tt = np.array(tt)
#     yy = np.array(yy)
#     ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(np.fft.fft(yy))
#     guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = np.std(yy) * 2.**0.5
#     guess_offset = np.mean(yy)
#     guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

#     def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*np.pi)
#     fitfunc = lambda t: A * np.sin(w*t + p) + c
#     return {"amp":A, "omega":w, "phase":p, "offset":c, "freq":f, "period":1./f, "fitfunc":fitfunc, "maxcov":np.max(pcov), "rawres": (guess,popt,pcov)}
