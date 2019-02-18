import pandas as pd
import numpy as np
import scipy as sp

import utility_functions as fn

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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


def snapshotplot(df, column, start, stop):
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


def gap_check(df,gap_indicator):
    """
    a snippet i wrote that finds and prints the integer indices of gaps given by a gap indicator
    df: DataFrame with datetime[ns] index
    gap_indicator: str giving name of column indicating gaps using True or False
    """
    
    print('start | end')
    
    for i in range(len(df)):
        if (df['impute_ok'][i-1]==True) & (df['impute_ok'][i]==False):
            gap_start=i
            for j in range(i,len(df)):
                if df['impute_ok'][j]==True:
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
    
    
    
    
    
