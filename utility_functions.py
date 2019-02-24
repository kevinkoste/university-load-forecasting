import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def fit_sin(tt, yy):
    """
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
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp":A, "omega":w, "phase":p, "offset":c, "freq":f, "period":1./f, "fitfunc":fitfunc, "maxcov":np.max(pcov), "rawres": (guess,popt,pcov)}


def sinusoid_impute(df,):
    """
    Fits a sinusoidal curve to time-series data
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
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp":A, "omega":w, "phase":p, "offset":c, "freq":f, "period":1./f, "fitfunc":fitfunc, "maxcov":np.max(pcov), "rawres": (guess,popt,pcov)}


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


def plot_feature(df, column, start, stop):
    """
    Plots all features for a snapshot between two indices
    
    df: DataFrame with datetime[ns] index
    column: str with column name -> eventually remove and plot all features as subplots
    start: str of index label at which to start plot
    stop: str of index label at which to end plot
    """
    
    start_int = df.index.get_loc(start)
    stop_int = df.index.get_loc(stop)

    snapshot = df.iloc[start_int:stop_int]

    plot = plt.plot(snapshot.index, snapshot[column],'ro')

    plt.ylabel('generic ylabel')
    plt.xlabel('Time')
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
    start_int = df.index.get_loc(start)
    stop_int = df.index.get_loc(stop)

    snapshot = df.iloc[start_int:stop_int]

    fig = plt.figure(figsize=(20, 5*len(snapshot.columns)),dpi= 80,facecolor='w')

    for i in range(1,len(snapshot.columns)+1):
        ax = fig.add_subplot(len(snapshot.columns),3,i)
        ax.title.set_text(snapshot.columns[i-1])
        ax.plot(snapshot[snapshot.columns[i-1]],style)
    return


def gap_indicator(df_in,gap_length):
    """
    Returns a bool DataFrame indicating gaps longer than gap_length hours in df_in
    
    df: DataFrame to be checked, with datetime[ns] index
    gap_length: int maximum allowable gap length in hours
    """
    fward = (df_in.notna()
                  .rolling(gap_length,min_periods=1)
                  .sum()
                  .astype(bool)
            )
    
    
    bward = (df_in.iloc[::-1]
                  .notna()
                  .rolling(gap_length,min_periods=1)
                  .sum()
                  .iloc[::-1]
                  .astype(bool)
            )
    
    return fward & bward
    

def gap_check(df,gap_indicator_column):
    """
    Prints integer indices of starts and ends of all gaps indicated by the gap_indicator_column
    
    df: DataFrame with datetime[ns] index
    gap_indicator_column: str giving name of the column that indicates gaps using True or False
    """
    
    print('start | end')
    
    for i in range(len(df)):
        if (df[gap_indicator_column][i-1]==True) & (df[gap_indicator_column][i]==False):
            gap_start=i
            for j in range(i,len(df)):
                if df[gap_indicator_column][j]==True:
                    gap_end=j
                    break
            print(gap_start, gap_end)
        


# def fourier_impute(XXX args)
#     """
#     all the code I tried to use to implement fourier transform :( need help desperately
#     """
    
#     initial = scaled['temp'].iloc[0:4799]

#     transform = fftpack.rfft(initial)

#     complete = np.append(transform,np.array([np.median(transform)]*43))

#     recons = fftpack.irfft(complete)

#     # complete = np.append(weather['temp'].iloc[0:4799],np.array([weather['temp'].iloc[0:4799].median()]*43))
#     # weather['temp'].iloc[4500:4799].median()
#     # transform = fftpack.rfft(complete)
#     # reconstructed = fftpack.irfft(transform)

#     plt.plot(range(len(recons[4700:4842])),recons[4700:4842])

#     plt.plot(range(len(initial[4700:4799])),initial[4700:4799])

#     plt.plot(range(len(scaled['temp'][4700:4900])),scaled['temp'][4700:4900])

#     # plt.plot(range(100),transform[0:100])



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
    
    
    
