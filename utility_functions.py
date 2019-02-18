import pandas as pd
import numpy as np
import scipy as sp

import utility_functions as fn

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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
    
    
    
    
    
    
    
    
