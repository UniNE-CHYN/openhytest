#    Copyright (C) 2022 by
#    Nathan Dutler <nathan.dutler@exquiro.ch>
#    Philippe Renard <philippe.renard@unine.ch>
#    Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#    All rights reserved.
#    MIT license.


"""
Preprocessing tool
==================
The openhytest preprocessing is a Python package for time series selection,
preprocessing, resampling, filtering and visualization.
License
-------
Released under the MIT license:
   Copyright (C) 2021 openhytest Developers
   Nathan Dutler <nathan.dutlern@exquiro.ch>
   Philippe Renard <philippe.renard@unine.ch>
   Bernard Brixel <bernard.brixel@erdw.ethz.ch>
"""
import pandas as pd
import numpy as np
import scipy.interpolate as ipo
from scipy import signal
import plotly.graph_objects as go
import plotly.express as px
#from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)


# *************************************************************
# ------------------Public functions---------------------------
# ---------------Preprocessing Functions:----------------------
# *************************************************************

class preprocessing():

    def __init__(self, df=None, der=None, npoints=30, bourdetder = 2, method = 'spline', xstart=None, xend=None, typefilter='moving', p=10, win_types=None, nval=None, idlog='linear', option='sample', Qmat = None):
        self.df = df
        self.der = der
        self.npoints = npoints
        self.bourdetder = bourdetder
        self.method = method
        self.xstart = xstart
        self.xend = xend
        self.typefilter = typefilter
        self.p = p
        self.win_types = win_types
        self.nval = nval
        self.idlog = idlog
        self.option = option
        self.Qmat = Qmat

    def header(self):
        """
        header reads the header of the pandas df dataframe
        """
        hd = self.df.head(1)
        self.hd = list(hd)

    def ldiff(self, df=None):
        """
        ldiff creates the logarithmic derivative with centered difference scheme.

        :param df:  pandas DF expects at two traces in the dataframe.
            The first column needs to be the time.
             i.e. df.t
            The second df trace needs to be sampled at the given
            time in the first column.
                i.e. df.s

        :return der: logarithmic derivative in pandas dataframe format with the same
            names given by the input df.

        :Examples:
            >>>  test = ht.preprocessing(df=df)
            >>>  test.ldiff()
        """
        if df is not None:
            self.df = df

        self.header()

        #Calculate the difference dx
        x = self.df[self.hd[0]].to_numpy()
        dx = np.diff(x)

        #calculate xd
        xd = np.sqrt(x[:-1]*x[1:])

        #Calculate the difference dy
        dy = np.diff(self.df[self.hd[1]])
        yd = xd*(dy/dx)
        dummy = np.array(np.transpose([xd, yd]))
        self.der = pd.DataFrame(dummy, columns=self.hd)

        return self.der

    def ldiff_plot(self, df=None):
        """
        ldiff_plot creates the plot with logarithmic derivative with centered
        difference scheme.

        :param df : expects two vectors with t and s

        :returns : plot inclusive legend

        """
        if df is not None:
            self.df = df

        self.ldiff()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.df[self.hd[0]], 
                y=self.df[self.hd[1]],
                marker=dict(
                    color='Red',
                    size=8,
                    line=dict(
                        color='Black',
                        width=1
                    ) 
                ), 
                name='Drawdown'))

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.der[self.hd[0]], 
                y=self.der[self.hd[1]],
                marker=dict(
                    color='Blue',
                    size=6,
                    symbol=203,
                    line=dict(
                        color='Black',
                        width=1
                    )
                ), 
                name='Derivative')) 
        fig.update_layout(xaxis_title="Time [s]", yaxis_title="p and p'")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log") 
        fig.show()


    def ldiffs(self, df=None, npoints=None):
        """
        ldiffs creates the logarithmic derivative of x with 1D linear interpolation of y

        :param df:  pandas DF expects two traces in the dataframe.
            The first column needs to be the time.
             i.e. df.t
            The second df trace needs to be sampled at the given
            time in the first column.
                i.e. df.s

        :param npoints: optional argument allowing to adjust the number of points
                used in the interpolation  ; default is 30

        :returns der: logarithmic derivative in pandas dataframe format with the same
            names given by the input df.

        Examples
        --------
            >>> test = ht.preprocessing(df=df)
            >>> test.ldiffs()
        """
        if df is not None:
            self.df = df

        if npoints is not None:
            self.npoints = npoints

        self.header()

        #interpolation xi and yi
        x = self.df[self.hd[0]].to_numpy()
        xi = np.logspace(np.log10(x[0]), np.log10(x[-1]),  num=self.npoints, endpoint=True, base=10.0, dtype=np.float64)

        yi = np.interp(xi, x, self.df[self.hd[1]].to_numpy())

        xd = xi[1:len(xi)-1]
        yd = xd*(yi[2:len(yi)]-yi[0:len(yi)-2])/(xi[2:len(xi)]-xi[0:len(xi)-2])

        dummy = np.array(np.transpose([xd, yd]))
        self.der = pd.DataFrame(dummy, columns=self.hd)
        return self.der


    def ldiffs_plot(self, df=None):
        """
        ldiffs_plot creates the plot with logarithmic derivative with 1D linear interpolation 

        :param df : expects two vectors with t and s

        :returns : plot inclusive legend

        """
        if df is not None:
            self.df = df

        self.ldiffs()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.df[self.hd[0]], 
                y=self.df[self.hd[1]],
                marker=dict(
                    color='Red',
                    size=8,
                    line=dict(
                        color='Black',
                        width=1
                    ) 
                ), 
                name='Drawdown'))

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.der[self.hd[0]], 
                y=self.der[self.hd[1]],
                marker=dict(
                    color='Blue',
                    size=6,
                    symbol=203,
                    line=dict(
                        color='Black',
                        width=1
                    )
                ), 
                name='Derivative')) 
        fig.update_layout(xaxis_title="Time [s]", yaxis_title="p and p'")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log") 
        fig.show()


    def ldiffb(self, df=None, bourdetder=None):
        """
        ldiffb creates the logarithmic derivative with Bourdet's formula.

        :param df:  pandas DF expects at least two traces in the dataframe.
            The first column needs to be the time.
                i.e. df.t
            The second df trace needs to be sampled at the given
            time in the first column.
                i.e. df.s

        :param bourdetder: optional argument allowing to adjust the distance between
               successive points to calculate the derivative. default is 2.

        :returns der: logarithmic derivative in pandas dataframe format with the same
            names given by the input df.
        """
        if df is not None:
            self.df = df

        if bourdetder is not None:
            self.bourdetder = bourdetder

        self.header()

        x = self.df[self.hd[0]].to_numpy()
        logx = np.log(x)
        dx = np.diff(logx)
        dx1 = dx[0:len(dx)-2*self.bourdetder]
        dx2 = dx[2*self.bourdetder-1:len(dx)-1]

        #Calculate the difference dy
        dy = np.diff(self.df[self.hd[1]].to_numpy())
        dy1 = dy[0:len(dy)-2*self.bourdetder]
        dy2 = dy[2*self.bourdetder-1:len(dy)-1]

        #xd and yd
        xd = np.array(x[self.bourdetder:len(self.df[self.hd[1]])-self.bourdetder-1])
        yd = (dx2*dy1/dx1+dx1*dy2/dx2)/(dx1+dx2)
        dummy = np.array(np.transpose([xd, yd]))
        self.der = pd.DataFrame(dummy, columns=self.hd)
        return self.der


    def ldiffb_plot(self, df=None):
        """
        ldiffb_plot creates the plot with logarithmic derivative with Bourdet's formula

        :param df : expects two vectors with t and s

        :returns : plot inclusive legend
        """
        if df is not None:
            self.df = df

        self.ldiffb()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.df[self.hd[0]], 
                y=self.df[self.hd[1]],
                marker=dict(
                    color='Red',
                    size=8,
                    line=dict(
                        color='Black',
                        width=1
                    ) 
                ), 
                name='Drawdown'))

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.der[self.hd[0]], 
                y=self.der[self.hd[1]],
                marker=dict(
                    color='Blue',
                    size=6,
                    symbol=203,
                    line=dict(
                        color='Black',
                        width=1
                    )
                ), 
                name='Derivative')) 
        fig.update_layout(xaxis_title="Time [s]", yaxis_title="p and p'")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log") 
        fig.show()


    def ldiffh(self, df=None):
        """
        ldiffh creates the logarithmic derivative with Horner formula.

        :param df:  pandas DF expects at least two traces in the dataframe.
            The first column needs to be the time.
                i.e. df.t
            The second df trace needs to be sampled at the given
            time in the first column.
                i.e. df.s

        :returns der: logarithmic derivative in pandas dataframe format with the same
            names given by the input df.

        """
        if df is not None:
            self.df = df

        self.header()

        #create the table x1,x2,x3 and y1,y2,y3
        x = self.df[self.hd[0]].to_numpy()
        x1 = np.array(x[:-2])
        x2 = np.array(x[1:-1])
        x3 = np.array(x[2:])
        xd = x2

        y = self.df[self.hd[1]].to_numpy()
        y1 = np.array(y[:-2])
        y2 = np.array(y[1:-1])
        y3 = np.array(y[2:])

        ################ to know what is what ##################
        #d1 = (log(t2./t1).*s3)./       (log(t3./t2).*log(t3./t1));
        #d2 = (log(t3.*t1./t2.^2).*s2)./(log(t3./t2).*log(t2./t1));
        #d3 = (log(t3./t2).*s1)./ (log(t2./t1).*log(t3./t1));

        #### d1 ####
        d1 = (np.log(x2/x1) * np.array(y3)) / (np.log(x3/x2)*np.log(x3/x1))

        #### d2 ####
        d2 = (np.log(x3*x1/x2**2) * y2) / (np.log(x3/x2)*np.log(x2/x1))

        #### d3 ####
        d3 = (np.log(x3/x2) * y1) / (np.log(x2/x1)*np.log(x3/x1))

        yd = d1+d2-d3
        dummy = np.array(np.transpose([xd, yd]))
        self.der = pd.DataFrame(dummy, columns=self.hd)
        return self.der

    def ldiffh_plot(self, df=None):
        """
        ldiffh_plot creates the plot with logarithmic derivative with Horner formula

        :param df : expects two vectors with t and s

        :returns : plot inclusive legend
        """
        if df is not None:
            self.df = df

        self.ldiffh()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.df[self.hd[0]], 
                y=self.df[self.hd[1]],
                marker=dict(
                    color='Red',
                    size=8,
                    line=dict(
                        color='Black',
                        width=1
                    ) 
                ), 
                name='Drawdown'))

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.der[self.hd[0]], 
                y=self.der[self.hd[1]],
                marker=dict(
                    color='Blue',
                    size=6,
                    symbol=203,
                    line=dict(
                        color='Black',
                        width=1
                    )
                ), 
                name='Derivative')) 
        fig.update_layout(xaxis_title="Time [s]", yaxis_title="p and p'")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log") 
        fig.show()


    def diagnostic(self, df=None, method=None):
        """
        diagnostic creates a diagnostic plot of the df
        (i.e. a log-log plot of the drawdown as a function of time together with its logarithmic derivative)

        :param df:  pandas series expects at least two traces in the dataframe.
            The first column needs to be the time.
                i.e. df.t
            The second df trace needs to be sampled at the given
            time in the first column.
                i.e. df.s

        :param method: optional argument allowing to select different methods of
            computation of the derivative

            'spline' for spline resampling in that case d is the number of points for the spline

            'direct' for direct derivation in that case the value provided in the variable d is not used

            'bourdet' for the Bourdet et al. formula in that case d is the lag distance used to compute the derivative

            'horner'  for the logarithmic derivative with Horne formula

            'norm' for data presentation of slug or pulse tests with normalised pressure

        :return: diagnostic plot inclusive legend
        """
        if df is not None:
            self.df = df

        if method is not None:
            self.method = method

        if self.method == 'spline':
            self.ldiffs_plot()
        elif self.method == 'direct':
            self.ldiff_plot()
        elif self.method == 'bourdet':
            self.ldiffb_plot()
        elif self.method == 'horner':
            self.ldiffh_plot()
        elif self.method == 'norm':
            self.norm_plot()
        else :
            print('ERROR: diagnostic(df,method)')
            print(' The method selected for log-derivative calculation is unknown')


    def hyclean(self, df=None):
        """
        hyclean: Take only the values that are not nan, finite and strictly positive time.

        :param df:  pandas DF expects at least two traces in the dataframe.
            The first column needs to be the time and the second the data trace.

        :return df: pandas series gives back the cleaned dataset
        """
        if df is not None:
            self.df = df

        self.header()
        df = self.df.replace([np.inf, -np.inf], np.nan)
        for i in range(1, len(self.hd)):
            self.df = df[df[self.hd[i]] >= 0]

        return self.df
    

    def hyselect(self, df=None, xstart=None, xend=None):
        """
        hyselect Select a part of a dataset strictly between xstart and xend

        :param df:  pandas DF expects at least two traces in the dataframe.
                The first column needs to be the time and the second the data trace.

        :param xstart: start of period, which will be selected
        :param xend: end of period, which will be selected

        :return df: pandas series gives back the selected dataset
        """
        if df is not None:
            self.df = df

        if xstart is not None:
            self.xstart = xstart

        if xend is not None:
            self.xend = xend

        self.header()

        mask = (self.df[self.hd[0]] > self.xstart) & (self.df[self.hd[0]] < self.xend)
        self.df = self.df.loc[mask]

        return self.df
    
    def hynorm(self, df=None, maximum_value=None, initial_value=None): 
        """
        hynorm normalizes the drawdown/recovery curve to the interval between [0, 1]
        It is most often used for pulse test data sets.
        
        It is recomended to give a maximum and intial value, if available, such 
        that the curve correlate with the origin curve. If no values are give the 
        normalization is done with the maximum and minimum value found in the data trace.
        

        :param df:  pandas DF expects at least two traces in the dataframe.
                The first column needs to be the time and the second the data trace.

        :param maximum_value: maximum value observed during the test
        :param initial_value: initial value at steady state conditions

        :return norm: pandas series gives back the normalized drawdown/recovery
        """
        
        if df is not None:
            self.df = df
            
        self.header() 

        t = self.df[self.hd[0]].to_numpy()

        if (maximum_value and initial_value) is not None:
            DP = maximum_value-initial_value
            s = (self.df[self.hd[1]].to_numpy() - initial_value) / DP
            dummy = np.array(np.transpose([t, s]))
            self.norm = pd.DataFrame(dummy, columns=self.hd)
        else:
            s = self.df[self.hd[1]].to_numpy()
            DP = np.amax(s) - np.amin(s)
            s1 = (s - np.amin(s)) / DP
            dummy = np.array(np.transpose([t, s1]))
            self.norm = pd.DataFrame(dummy, columns=self.hd)
 
        return self.norm


    def norm_plot(self, df=None, maximum_value=None, initial_value=None):
        """
        norm_plot creates the plot with normalised pressure using hynorm for normalisation of pressure

        :param df : expects two vectors with t and s

        :returns : plot inclusive legend
        """
        if df is not None:
            self.df = df

        self.hynorm(maximum_value=maximum_value, initial_value=initial_value)
        self.ldiffs()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.df[self.hd[0]], 
                y=self.df[self.hd[1]],
                marker=dict(
                    color='Red',
                    size=8,
                    line=dict(
                        color='Black',
                        width=1
                    ) 
                ), 
                name='Drawdown'))

        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=self.der[self.hd[0]], 
                y=np.abs(self.der[self.hd[1]].to_numpy()),
                marker=dict(
                    color='Blue',
                    size=6,
                    symbol=203,
                    line=dict(
                        color='Black',
                        width=1
                    )
                ), 
                name='Derivative')) 
        fig.update_layout(xaxis_title="Time [s]", yaxis_title="Normalised pressure")
        fig.update_xaxes(type="log")
        fig.update_yaxes(range=[0, 1])
        fig.show()


    def hyfilter(self, df=None, typefilter=None, p=None, win_types=None):
        """
        Filters a signal to reduce noise.

        Time steps need to be equally spaced for the butterworth filter.
        Unequally-spaced time steps can be used for moving average filter (not recommended).

        :param df:  pandas DF expects at least two traces in the dataframe.
                    The first column needs to be the time and the second the df trace.

        :param typefilter: allows to select the type of filter

        'butter3' for the Butterworth filter with filter order 2 or
        'butter5' for the Butterworth filter with filter order 4.
        The Butterworth filters high frequency components in the signal.
        It is very sensitive of outliers. Therefore the Butterworth filter is
        more appropriate for noisy signals without outliers, the moving average is
        recommended to filter very irregular signals.

        'moving' for the moving average (Default option).
        The moving average is less sensitive of outliers but at the same time
        less smooth using a centered windows.

        :param p: parameter that depends on the type of filter that has been chosen.

          for the moving average
               p = size of the moving window in number of points
                   by default it is equal to 11. It has to be an odd number.

          for the Butterworth filter
                p = period of the cutoff frequency in number of measurements
                    by default it is equal to 10

        :param win_types: is only defined for 'moving' filter. It defines the type of
                weighting window: boxcar, triang, blackman, hamming
                (see pandas DF rolling command for more options and information)

        :return df: pandas series is gives back with the name self.filtered

        :examples:
            >>>  self.hyfilter(typefilter='moving', p=13)
            >>>  self.hyfilter(typefilter='moving', p=8, win_types='triang')
            >>>  self.hyfilter(typefilter='butter3', p=10)
        """
        if df is not None:
            self.df = df

        if typefilter is not None:
            self.typefilter = typefilter

        if p is not None:
            self.p = p

        if win_types is not None:
            self.win_types = win_types

        self.header()

        if self.typefilter == 'moving':
            if self.p == 10:
                self.p = 11
            elif self.p % 2 == 0:
                print('ERROR: Make sure, that the size of the moving filter has an odd number.')
            xs = self.df[self.hd[0]]
            ys = self.df.iloc[:,1].rolling(window=self.p, center=True, win_type=self.win_types).mean()
            dummy = np.array(np.transpose([xs, ys]))
            self.filtered = pd.DataFrame(dummy, columns=self.hd)
            return self.filtered
        elif self.typefilter == 'butter3':
            xs = self.df[self.hd[0]]
            ts = self.df[self.hd[0]][1]-self.df[self.hd[0]][0]
            fs = 2/ts
            fc = 1.5*fs/self.p
            b, a = signal.butter(3, fc, 'low')
            ys = signal.filtfilt(b, a, self.df[self.hd[1]])
            dummy = np.array(np.transpose([xs, ys]))
            self.filtered = pd.DataFrame(dummy, columns=self.hd)
            return self.filtered
        elif self.typefilter == 'butter5':
            xs = self.df[self.hd[0]]
            ts = self.df[self.hd[0]][1]-self.df[self.hd[0]][0]
            fs = 2/ts
            fc = 1.5*fs/p
            b, a = signal.butter(5, fc, 'low')
            ys = signal.filtfilt(b, a, self.df[self.hd[1]])
            dummy = np.array(np.transpose([xs, ys]))
            self.filtered = pd.DataFrame(dummy, columns=self.hd)
            return self.filtered
        else:
            print('ERROR: The function hyfilter does not know the filter type.')

        return self.df


    

    def hysampling(self, df=None, nval=None, idlog='log', option='interp1Dlin'):
        """
        Sample a signal at regular intervals

        :param df: expects at least two traces in the dataframe.
                   The first column takes time as an input. The second column takes
                   the df trace.
        :param xs,ys: sampled signal
        :param nval: number of values of the output signal

            idlog: allows to select the type of spacing of the samples
                idlog  = 'linear' = Default value = linear scale
                idlog  = 'log'    = logarithmic scale

            option: allows to define if the points must be interpolated
                option = 'sample' = Default value
                                  = only points from the data set are taken, can lead to smaller number of given nval points
                option = 'interp1Dlin' = creates points by linear spline interpolation
                option = 'interp1Dsquare' = creates points by quadratic interpolation
                option = 'interp1Dcubic' = creates points by cubic interpolation
                
        :return sampeld: gives the resampled time, t interpolated drawdown, s 

        """
        if df is not None:
            self.df = df

        if nval is not None:
            self.nval = nval

        self.header()

        x = self.df[self.hd[0]].to_numpy()
        y = self.df[self.hd[1]].to_numpy()

        xs = np.empty(nval)
        ys = np.empty(nval)
            

        # logarithmic sampling
        if idlog == 'log': #WORKS
            xs = np.logspace(np.log10(x[0]), np.log10(x[-1]), nval)

        # linear sampling
        elif idlog == 'linear': #WORKS
            xs = np.linspace(x[0], x[-1], nval)

        else :
            print('')
            print('SYNTAX ERROR: hysampling: the idlog is incorrect.')
            print('')

        if option == 'sample': 
            
            if nval > len(x):
                print('')
                print('SYNTAX ERROR: nval is larger than the number of data points')
                print('')

            for i in range(0, nval):

                # find sampling location
                dist = np.sqrt(np.power(x - xs[i], 2)) # I think this should be 2 and not 3
                mn = np.nanmin(dist)

                # get index
                j = np.asarray(np.where(dist == mn))
                j.resize(1)  # avoids having multiple elements if more than one min value exists

                # assign index to sample array
                xs[i] = x[j]
                ys[i] = y[j]

                # remove duplicates
                xs_nodup, index_xs = np.unique(xs, return_index=True)
                ys_nodup = ys[index_xs]
                dummy = np.array(np.transpose([xs_nodup, ys_nodup]))
                self.sampeld = pd.DataFrame(dummy, columns=self.hd)
            return self.sampeld

        # sample interpolated 'y' data points
        elif option == 'interp1Dlin':
            f_interp = ipo.interp1d(x, y, kind='slinear', fill_value="extrapolate")
            ys = f_interp(xs)
            ys = np.asarray(ys, dtype='float')
            dummy = np.array(np.transpose([xs, ys]))
            self.sampeld = pd.DataFrame(dummy, columns=self.hd)
            return self.sampeld
        elif option == 'interp1Dsquare':
            f_interp = ipo.interp1d(x, y, kind='quadratic', fill_value="extrapolate")
            ys = f_interp(xs)
            ys = np.asarray(ys, dtype='float')
            dummy = np.array(np.transpose([xs, ys]))
            self.sampeld = pd.DataFrame(dummy, columns=self.hd)
            return self.sampeld
        elif option == 'interp1Dcubic':
            f_interp = ipo.interp1d(x, y, kind='cubic', fill_value="extrapolate")
            ys = f_interp(xs)
            ys = np.asarray(ys, dtype='float')
            dummy = np.array(np.transpose([xs, ys]))
            self.sampeld = pd.DataFrame(dummy, columns=self.hd)
            return self.sampeld

        else:
            print('')
            print('SYNTAX ERROR: hysampling the parameter option is incorrect.')
            print('')
            return None
        

    def drawdownUnconf(self, s, b):
        """
        Computes the corrected drawdown for tests in unconfined aquifers where the drawdown
        is substantial (>30%) compared with the aquifer saturated thickness.
         
        :param s: numpy array, drawdown
        :param b: aquifer saturated thickness
         
        :Reference: Hantush, M.S. (1956),  Analysis of data from pumping tests in
        leaky aquifers. Transactions, American Geophysical Union, 36:702-14.
        """
        s_corr = (s[0]-s)-((s[0]-s)**2/(2*b))
        return s_corr     
    
         
    def fracflowdim(self, df=None):
        """
        Computes the time evolution of apparent flow dimensions (n) for pressure 
        transients whose late time pressure derivative slope is strictly
        positive (i.e. subradial flow regimes, where 1<n<2).
        
        :param df:  pandas dataframe with two vectors, time and drawdown
        :return dim: calcualtes the apparent flow dimension for the supplied dataset
        
        :Reference: Le Borgne, T., Bour, O., De Dreuzy, J-R., Davy, P. Touchard, P..
        (2004), Equivalent mean flow models for fractured aquifers: Insights from a 
        pumping tests scaling interpretation. WRR, 40, W03512, doi:10.1029/2003WR002436. 
        and Brixel, B., Klepikova, M., Lei, Q., Roques, C., Jalali, M., Krietsch, H., 
        Loew, S. (2020), Tracking fluid flow in shallow crustal fault zones: 2. Insights 
        from cross-hole forced flow experiments in damage zones. JGR Solid Earth, 125, 
        4, doi.org/10.1029/2019JB018200
        """
        if df is not None:
            self.df = df

        self.header()

        # removes all NaN and finite, strictly positive
        self.hyclean()

        x = self.df[self.hd[0]][1:]
         
        # compute flow dimension
        y = np.multiply(3, (1 - np.divide(([np.log10(x) - np.log10(self.df[self.hd[1]][i - 1]) for i, x in enumerate(self.df[self.hd[1]]) if i > 0]), [
            np.log10(x) - np.log10(self.df[self.hd[0]][i - 1]) for i, x in enumerate(self.df[self.hd[0]]) if i > 0])))
        dummy = np.array(np.transpose([x, y]))
        self.dim = pd.DataFrame(dummy, columns=self.hd)
        return self.dim

    def birsoy_time(self, df=None, Qmat=None):
        """
        Calculates the equivalent time of Birsoy and Summers (1981) solution.

        :param df: needs a pandas dataframe with vector t and s
        :param Qmat: needs a pandas dataframe with vector t and q
          size = nb of lines = nb of pumping periods
          two colums
                column t = time (since the beginning of the test at which the period ends
                column q = flow rate during the period

        :return birsoy: equivalent Birsoy and Summers time, t and s 
        
        :Reference: Birsoy, Y.K. and Summers, W.K. (1980), Determination of 
        Aquifer Parameters from Step Tests and Intermittent Pumping Data. 
        Groundwater, 18: 137-146. doi:10.1111/j.1745-6584.1980.tb03382.x
        """
        if df is not None:
            self.df = df

        self.header()

        if Qmat is not None:
            self.Qmat = Qmat

        if np.size(self.Qmat.t) < 2:
            print('Warning - birsoy_time function: The Qmat contains only 1 line')

        pe = np.zeros(np.size(self.df[self.hd[1]]))
        for i in range(np.size(self.Qmat.t), 0, -1):
            j = self.df.index[self.df.t.le(self.Qmat.t[i-1])]
            pe[j] = i-1
        lq = self.Qmat.q[pe].to_numpy()
        
        dq = np.diff(self.Qmat.q)
        dq = np.insert(dq, 0, self.Qmat.q[0], axis = 0)

        st = [0, self.Qmat.q[:-1]]
        st = self.Qmat.t[:-1].to_numpy()
        st = np.insert(st, 0, 0, axis=0)
        t  = np.ones(np.size(self.df.t.to_numpy()))
        for j in range(0, np.size(self.df.t.to_numpy()), 1):
            for i in range(0, int(pe[j])+1, 1):
                t[j] = t[j]* (self.df.t[j] - st[i]) ** (dq[i]/ self.Qmat.q[pe[j]])

        ind = np.argsort(t)
        t = t[ind]
        s = self.df.s.to_numpy() 
        s = s[ind] / lq[ind]     
        dummy = np.array(np.transpose([t, s]))
        self.birsoy = pd.DataFrame(dummy, columns=self.hd)    
        
        return self.birsoy
    
    
    def agarwal(self, df=None, pinj=None, Qmat=None, agarwal=None):
        """
        Computes equivalent Agarwal (1980) time and recovery transformation for 
        pressure recovery tests to apply and use drawdown models from pumping tests.
        Agarwal has shown in 1980 that recovery test can be interpreted with 
        the same solutions than pumping test if one interprets the residual
        drawdown sr = s(end of pumping) - s(t) as a function of an equivalent
        time that is computed from the pumping rates history. The theory is 
        based on the superposition principle.

        :param df: needs a pandas dataframe with vector t and s
                column t = vector containing the time since the beginning of the recovery
                column s = the residual drawdown is defined as follows:
                    sr = s(end of pumping) - s(t)
                    It is equal to 0 when the the pumping stops and it increases progressively when the aquifer recovers its equilibrium.
        
        
        :param Qmat: can be for a single rate pumping test 
            or can handle a pandas dataframe with vector t time and q flow rate during the period
          
        :return agrawal: equivalent Agarwal (1980) time and drawdown, t and s as pandas dataframe

        :Reference: Agarwal, R.G., 1980. A new method to account for producing
        time effects when drawdown type curves are used to analyse pressure 
        buildup and other test data. Proceedings of the 55th Annual Fall 
        Technical Conference and Exhibition of the Society of Petroleum 
        Engineers. Paper SPE 9289  
        """        
        if df is not None:
            self.df = df
        
        if pinj is not None:
            self.pinj = pinj
        else:
            self.pinj = self.df.s[0]

        self.header()

        if Qmat is not None:
            self.Qmat = Qmat
        
        if np.size(Qmat) == 1: 
            t = self.Qmat*self.df.t.to_numpy() / (self.Qmat + self.df.t.to_numpy())
          
        elif isinstance(self.Qmat, pd.DataFrame): 
            tp = self.Qmat.t.to_numpy()
            qp = self.Qmat.q.to_numpy()
            t = tp[-2]/(self.df.t.to_numpy()+tp[-2]) ** (qp[0]/(qp[-2]-qp[-1]))
            for j in range(1,np.size(tp)-1,1):
                t = t * ((tp[-2]-tp[j-1])/ (self.df.t.to_numpy()+ tp[-2]-tp[j-1])) ** ((qp[j]-qp[j-1])/(qp[-2]-qp[-1]))
            t = t * self.df.t.to_numpy()
        else:
            print('Qmat needs to be a scalar or pandas dataframe')
         
        t, indices = np.unique(t, return_index=True)
        s = self.df.s[0] - self.df.s[indices]
        dummy = np.array(np.transpose([t, s]))
        self.agarwal = pd.DataFrame(dummy, columns=self.hd) 
        return self.agarwal
        
