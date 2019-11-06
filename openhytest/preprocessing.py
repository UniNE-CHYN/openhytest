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


# *************************************************************
# ---------------Preprocessing Functions:----------------------
# *************************************************************

def ldiff(x,y):
    """
    ldiff creates the logarithmic derivative with centered difference scheme.
    -----
    x:  pandas series
        In hydrotests x expects a single time serie.
    
    y:  pandas series
        In hydrotests y expects single drawdowns.
    
    Returns
    -------
    xd, yd
        logarithmic derivative in pandas time series format
    
    Examples
    --------
       >>> xd, yd = ht.ldiff(x,y)
    """
    
    #Calculate the difference
    dx = np.diff(x)
    dy = np.diff(y)

    #calculate xd
    xd = np.sqrt(x[0:-1]*x[1:])
    
    #calculate yd
    yd = xd*(dy/dx)

    return pd.Series(xd), pd.Series(yd)


def ldiff_plot(x,y):
    """
    ldiff_plot creates the plot with logarithmic derivative with centered 
    difference scheme.
    ---------
    x:  pandas series
        In hydrotests x expects a single time serie.
    
    y:  pandas series
        In hydrotests y expects single drawdowns.
    
    Returns
    -------
    plot
    
    Examples
    --------
       >>> ht.ldiff_plot(x,y)
    """
    
    xd,yd = ldiff(x,y)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown and log derivative')
    
    ax2.loglog(x, y, c='b', marker = 'o', linestyle = '', label = 'drawdown')
    ax2.scatter(xd, yd, c='r', marker = 'x', label = 'derivative')
    ax2.grid(True)

    ax2.legend()
    
    plt.show()


