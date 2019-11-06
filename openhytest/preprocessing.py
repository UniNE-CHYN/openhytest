#    Copyright (C) 2019 by
#    Philippe Renard <philippe.renard@unine.ch>$
#    Nathan Dutler <nathan.dutler@unine.ch>
#    Bernard Brixel <bernard.brixel@ethz.ch>
#    All rights reserved.
#    MIT license.


"""
Preprocessing tool
========
The openhytest preprocessing is a Python package for time series selection, 
reprocesssing, resampling, filtering and visualization. 
License
-------
Released under the MIT license:
   Copyright (C) 2019 openhytest Developers
   Philippe Renard <philippe.renard@unine.ch>
   Nathan Dutler <nathan.dutlern@unine.ch>
   Bernard Brixel <bernard.brixel@ethz.ch>
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openhytest as ht


# *************************************************************
# ---------------Preprocessing Functions:----------------------
# *************************************************************

def ldiff(data):
    """
    ldiff creates the logarithmic derivative with centered difference scheme.
    -----
    data:  pandas series expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data traces needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2

    Returns
    -------
    derivative 
        logarithmic derivative in pandas time series format with the same
        names given by the input data
       
    
    Examples
    --------
        >>> derivative = ht.ldiff(data)   
    """
    df = data.head(0)
    df = list(df)
       
    #Calculate the difference dx
    x = data[df[0]].to_numpy()
    dx = np.diff(x)
    #calculate xd
    xd = np.sqrt(x[0:-1]*x[1:])
    
    #Calculate the difference dy
    for i in range(1,len(df)):
        dy = np.diff(data[df[i]]) 
        #calculate yd
        yd = xd*(dy/dx)
        if i == 1:
            der = np.array(np.transpose([xd,yd]))
        else:
            der = np.c_[der, np.transpose(yd)] 
            
    return pd.DataFrame(der, columns = df)


def ldiff_plot(data):
    """
    ldiff_plot creates the plot with logarithmic derivative with centered 
    difference scheme.
    ---------
    data : expects at least two vectors with x and y1, y2, y3,...
        
    Returns
    -------
    plot inclusive legend 
    
    Examples
    --------
       >>> ht.ldiff_plot(data)
    """
    
    derivative = ht.ldiff(data)
    df = data.head(0)
    df =  list(df)
    
    ax = data.plot(x = df[0], y=df[1:], loglog=True, marker='o', linestyle = '', colormap='jet')
    derivative.plot(x = df[0], y=df[1:], marker='x', loglog=True, linestyle = '', colormap='jet', ax=ax, grid=True)
    ax.set(xlabel='Time', ylabel='Drawdown and log derivative')
    
    ax.legend()
    
    return ax

    
    
    
    


