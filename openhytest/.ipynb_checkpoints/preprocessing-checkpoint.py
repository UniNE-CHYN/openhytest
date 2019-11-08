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
reprocessing, resampling, filtering and visualization. 
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
import openhytest as ht
from scipy.interpolate import UnivariateSpline

# *************************************************************
# ---------------Preprocessing Functions:----------------------
# *************************************************************

def ldiff(data):
    """
    ldiff creates the logarithmic derivative with centered difference scheme.
    -----
    data:  pandas DF expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2

    Returns
    -------
    derivative 
        logarithmic derivative in pandas dataframe format with the same
        names given by the input data.
       
    
    Examples
    --------
        >>> derivative = ht.ldiff(data)   
    """
    global der
    df = data.head(0)
    df = list(df)
       
    #Calculate the difference dx
    x = data[df[0]].to_numpy()
    dx = np.diff(x)
    #calculate xd
    xd = np.sqrt(x[0:-1]*x[1:])
    
    #Calculate the difference dy
    for i in range(1, len(df)):
        dy = np.diff(data[df[i]]) 
        #calculate yd
        yd = xd*(dy/dx)
        if i == 1:
            der = np.array(np.transpose([xd, yd]))
        else:
            der = np.c_[der, np.transpose(yd)] 
            
    return pd.DataFrame(der, columns=df)


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
    
    ax = data.plot(x=df[0], y=df[1:], loglog=True, marker='o', linestyle='', colormap='jet')
    derivative.plot(x=df[0], y=df[1:], marker='x', loglog=True, linestyle='', colormap='jet', ax=ax, grid=True)
    ax.set(xlabel='Time', ylabel='Drawdown and log derivative')
    
    ax.legend()


def ldiffs(data, npoints = 20):
    '''
    ldiffs creates the logarithmic derivative with spline functions.
    ------
    data:  pandas DF expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2
            
    npoints = optional argument allowing to adjust the number of points 
            used in the Spline

    Returns
    -------
    derivative 
        logarithmic derivative in pandas dataframe format with the same
        names given by the input data.
           
    Examples
    --------
        >>> derivative = ht.ldiffs(data, [d])
    '''
    global der
    df = data.head(0)
    df = list(df)

    #interpolation xi and yi
    x = data[df[0]].to_numpy() 
    xi = np.logspace(np.log10(x[0]), np.log10(x[len(x)-1]),  num=npoints, endpoint=True, base=10.0, dtype=np.float64)
    
    for i in range(1, len(df)):
        #changing k & s affects the interplolation
        spl = UnivariateSpline(x,np.array(data[df[i]].to_numpy()), k=5, s=0.0099)
        yi = spl(xi)

        xd = xi[1:len(xi)-1]
        yd = xd*(yi[2:len(yi)]-yi[0:len(yi)-2])/(xi[2:len(xi)]-xi[0:len(xi)-2])       
        
        if i == 1:      
            der = np.array(np.transpose([xd, yd]))
       
        else:
            der = np.c_[der, np.transpose(yd)] 
    
    return pd.DataFrame(der, columns=df)


def ldiffs_plot(data, npoints=20):
    """
    ldiffs_plot creates the plot with logarithmic derivative using slpine functions.
    ---------
    data : expects at least two vectors with x and y1, y2, y3,...
 
    npoints = optional argument allowing to adjust the number of points 
            used in the Spline
            
    Returns
    -------
    plot inclusive legend
    
    Examples
    --------
       >>> ht.ldiffs_plot(data,[d]])
    """
    derivative = ht.ldiffs(data, npoints)
    df = data.head(0)
    df = list(df)
    
    ax = data.plot(x=df[0], y=df[1:], loglog=True, marker='o', linestyle='', colormap='jet')
    derivative.plot(x=df[0], y=df[1:], marker='x', loglog=True, linestyle='', colormap='jet', ax=ax, grid=True)
    ax.set(xlabel='Time', ylabel='Drawdown and log derivative')
    
    ax.legend()
    
    
def ldiffb(data, d=2):
    '''
    ldiffb creates the logarithmic derivative with Bourdet's formula.
    ------
    data:  pandas DF expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2
            
    d = optional argument allowing to adjust the distance between 
           successive points to calculate the derivative.

    Returns
    -------
    derivative 
        logarithmic derivative in pandas dataframe format with the same
        names given by the input data.
           
    Examples
    --------
        >>> derivative = ht.ldiffb(data, [d])  
    '''
    global der
    df = data.head(0)
    df = list(df)    
    
    x = data[df[0]].to_numpy()
    logx = np.log(x)
    dx = np.diff(logx)
    dx1 = dx[0:len(dx)-2*d+1]
    dx2 = dx[2*d-1:len(dx)] 

    #Calculate the difference dy
    for i in range(1, len(df)):
        dy = np.diff(data[df[i]])
        dy1 = dy[0:len(dx)-2*d+1]
        dy2 = dy[2*d-1:len(dy)]

        #xd and yd
        xd = np.array(x[2:len(data[df[i]])-2])
        yd = (dx2*dy1/dx1+dx1*dy2/dx2)/(dx1+dx2)
    
        if i == 1:      
            der = np.array(np.transpose([xd,yd]))
       
        else:
            der = np.c_[der, np.transpose(yd)] 
    
    return pd.DataFrame(der, columns=df)
    

def ldiffb_plot(data, d=2):
    """
    ldiffb_plot creates the plot with logarithmic derivative using Bourdet's  formula.
    ---------
    data : expects at least two vectors with x and y1, y2, y3,...
 
    d = optional argument allowing to adjust the distance between 
           successive points to calculate the derivative.
            
    Returns
    -------
    plot inclusive legend
    
    Examples
    --------
       >>> ht.ldiffb_plot(data,[d]])
    """
    derivative = ht.ldiffb(data, d)
    df = data.head(0)
    df = list(df)
    
    ax = data.plot(x=df[0], y=df[1:], loglog=True, marker='o', linestyle='', colormap='jet')
    derivative.plot(x=df[0], y=df[1:], marker='x', loglog=True, linestyle='', colormap='jet', ax=ax, grid=True)
    ax.set(xlabel='Time', ylabel='Drawdown and log derivative')
    
    ax.legend()
    
    
def ldiffh(data):
    '''
    ldiffh creates the logarithmic derivative with Horner formula.
    ------
    data:  pandas DF expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2
            
    Returns
    -------
    derivative 
        logarithmic derivative in pandas dataframe format with the same
        names given by the input data.
           
    Examples
    --------
        >>> derivative = ht.ldiffh(data, [d])  
    '''
    global der
    df = data.head(0)
    df = list(df)
    
    #create the table t1,t2,t3 and s1,s2,s3
    endt = len(data[df[0]])
    
    
    x = data[df[0]].to_numpy()
    x1 = np.array(x[0:endt-2])
    x2 = np.array(x[1:endt-1])
    x3 = np.array(x[2:endt])
    xd = x2
    
    for i in range(1, len(df)):
        ends = len(data[df[i]])
        y = data[df[i]].to_numpy()
        y1 = np.array(y[0:ends-2])
        y2 = np.array(y[1:ends-1])
        y3 = np.array(y[2:ends])

        ################ to know what is what ##################
        #d1 = (log(t2./t1).*s3)./       (log(t3./t2).*log(t3./t1));
        #d2 = (log(t3.*t1./t2.^2).*s2)./(log(t3./t2).*log(t2./t1));
        #d3 = (log(t3./t2).*s1)./ (log(t2./t1).*log(t3./t1));      
        
        #### d1 ####
    
        #log(t2/t1)*s3
        D1_part1 = np.log(x2/x1) * np.array(y3)
    
        #log(t3/t2)*log(t3/t1)
        D1_part2 = np.log(x3/x2)*np.log(x3/x1)
        d1 = D1_part1/D1_part2
    
        #### d2 ####
        
        #logt3t1t2 * s2
        D2_part1 = np.log(x3*x1/x2**2) * y2
    
        #log(t3/t2)*log(t2/t1)
        D2_part2 = np.log(x3/x2)*np.log(x2/x1)
        d2 = D2_part1 / D2_part2
    
        #### d3 ####
        
        #logt3/t2 * s1
        D3_part1 = np.log(x3/x2) * y1
    
        #log(t2/t1)*log(t3/t2)
        D3_part2 = np.log(x2/x1)*np.log(x3/x1)
        d3 = D3_part1 / D3_part2
    
        yd = d1+d2-d3
        
        if i == 1:      
            der = np.array(np.transpose([xd,yd]))
       
        else:
            der = np.c_[der, np.transpose(yd)]         
    
    return pd.DataFrame(der, columns=df)

    
def ldiffh_plot(data):
    """
    ldiffh_plot creates the plot with logarithmic derivative using Horner formula.
    -----------
    data : expects at least two vectors with x and y1, y2, y3,...
 

    Returns
    -------
    plot inclusive legend
    
    Examples
    --------
       >>> ht.ldiffh_plot(data)
    """
    derivative = ht.ldiffh(data)
    df = data.head(0)
    df = list(df)
    
    ax = data.plot(x=df[0], y=df[1:], loglog=True, marker='o', linestyle='', colormap='jet')
    derivative.plot(x=df[0], y=df[1:], marker='x', loglog=True, linestyle='', colormap='jet', ax=ax, grid=True)
    ax.set(xlabel='Time', ylabel='Drawdown and log derivative')
    
    ax.legend()   
    
def diagnostic(data, method = 'spline'):
    '''
    diagnostic creates a diagnostic plot of the data
    ----------
    (i.e. a log-log plot of the drawdown as a function of time together with its logarithmic derivative)
    
    data :  pandas series expects at least two traces in the dataframe.
        The first column needs to be the time.
         i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column. 
            i.e. data.s, data.s2    
    
    method :    optional argument allowing to select different methods of 
        computation of the derivative
        
    'spline' for spline resampling
    in that case d is the number of points for the spline
    
    'direct' for direct derivation
    in that case the value provided in the variable d is not used
    
    'bourdet' for the Bourdet et al. formula
    in that case d is the lag distance used to compute the derivative
    
    'horner'  for the logarithmic derivative with Horne formula
    
    Returns
    -------
    plot inclusive legend   

    Example: 
    -------
        diagnostic(data) 
        diagnostic(data,'horner')

        '''
    if method == 'spline':
        ldiffs_plot(data)
    elif method == 'direct':
        ldiff_plot(data)
    elif method == 'bourdet':
        ldiffb_plot(data)
    elif method == 'horner':
        ldiffh_plot(data)
    else : 
        print('ERROR: diagnostic(data,method)')
        print(' The method selected for log-derivative calculation is unknown')


def hyclean(data):
    '''
    hyclean Take only the values that are not nan, finite and strictly positive time.
    ------
    data:  pandas DF expects at least two traces in the dataframe.
        The first column needs to be the time.
        i.e. data.t
        The second and following data trace needs to be sampled at the given
        time in the first column.
        i.e. data.s, data.s2
    
    Returns
    -------
    data
        pandas series gives back the cleaned dataset
        
    Examples
    --------
        >>>  data = ht.hyclean(data)
    '''
    df = data.head(0)
    df = list(df)
    data = data.replace([np.inf, -np.inf], np.nan)
    for i in range(1, len(df)):
        data = data[data[df[i]] >= 0] 

    return data

