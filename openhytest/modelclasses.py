#    Copyright (C) 2019 by
#    Nathan Dutler <nathan.dutler@unine.ch>
#    Philippe Renard <philippe.renard@unine.ch>
#    Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#    All rights reserved.
#    MIT license.


"""
Analytical model classes
**************************

The different analytical model classes are implemented to fit the observations given as dataframe in time (t) and drawdown (s).

License
---------
Released under the MIT license:
   Copyright (C) 2019 openhytest Developers
   Nathan Dutler <nathan.dutlern@unine.ch>
   Philippe Renard <philippe.renard@unine.ch>
   Bernard Brixel <bernard.brixel@erdw.ethz.ch>
   
"""

import numpy as np
from scipy.special import expn as E1
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import mpmath as mp


# Utilities

def thiem(q, s1, s2, r1, r2):
    """
    Calculate the transmissivity with Thiem (1906) solution
    
    The Thiem method requires to know the pumping rate and the drawdown in
    two observation wells located at different distances from the pumping
    well.
    
    :param q: pumping rate, m3/s
    :param s1: drawdown in piezometer 1, m
    :param s2: drawdown in piezometer 2, m
    :param r1: distance piezometer 1 to pumping well, m
    :param r2: distance piezometer 2 to pumping well, m

    :return T: transmissivity, m2/s
        
    :Data set: The data set for this example comes from the following reference: 
    Kruseman and de Ridder (1994), Analysis and Evaluation of Pumping Test
    Data. International Institute for Land Reclamation and Improvement,
    Wageningen. The Netherlands. 377 pp.
    Data set from table 3.2 pp. 56-60
    The result found by Kruseman and de Rider was: T = 4.5 e-3 m2/s

    :Example:
    >>> q=0.00912
    >>> r1=0.8  
    >>> r2=30     
    >>> s1=2.236  
    >>> s2=1.088 
    >>> T=ht.thiem(q,s1,s2,r1,r2) 
    """
    return q / (2 * np.pi * (s1 - s2)) * np.log(r2 / r1);


def goodman_discharge(l, T, r0):
    """
    Computes discharge rate for a well close to a constant head boundary
    
    The flow field is a half infinite space with a constant head boundary.
    The aquifer is supposed to be confined, homogeneous and the well fully
    penetrating the aquifer.
    
    The calculation is based on the Dupuit's solution with an image well.
    This equation is also known as the Goodman formula (1965).
    
    :param l:   Distance to the hydraulic boundary m
    :param T:   Transmissivity m^2/s
    :param r0:  radius of the well m
    :return q:  flow rate m^3/s
    
    :Reference: Goodman, R., 1965. Groundwater inflows during tunnel driving.
    Engineering Geology, 2(2): 39-56. 
    """
    return 2 * np.pi * T * l / (np.log(2 * l / r0))


def get_logline(df):
    logt = np.log10(df.t).values
    Gt = np.array([logt, np.ones(logt.shape)])
    p = np.linalg.inv(Gt.dot(Gt.T)).dot(Gt).dot(df.s)
    p[1] = 10 ** (-p[1] / p[0])
    return p


# Parent generic class

class AnalyticalInterferenceModels():
    def __init__(self, Q=None, r=None, Rd=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd

    def _dimensionless_time(self, p, t):
        """
        Calculates dimensionless time
        """
        return (t / (0.5628 * p[1])) * 0.25

    def _dimensional_drawdown(self, p, sd):
        """
        Calculates the dimensional drawdown
        """
        return (sd * p[0] * 2) / 2.302585092994046

    def _laplace_drawdown(self, td, option='Stehfest'):  # default stehfest
        """
        Alternative calculation with Laplace inversion
        
        :param td:      dimensionless time
        :param x:       dummy parameter for inversion
        :param option:  Stehfest (default, dps=10, degree=16), dehoog
        :return sd:     dimensionless drawdown
        """        
        return list(map(lambda x: mp.invertlaplace(self.dimensionless_laplace, x, method=option, dps=10, degree=16), td))

    def _laplace_drawdown_derivative(self, td, option='Stehfest'):  # default stehfest
        """
        Alternative calculation with Laplace inversion
        
        :param td:      dimensionless time
        :param x:       dummy parameter for inversion
        :param option:  Stehfest (default, dps=10, degree=16), dehoog
        :return dd:     dimensionless drawdown derivative
        """        
        return list(map(
            lambda x: mp.invertlaplace(self.dimensionless_laplace_derivative, x, method=option, dps=10, degree=16), td))

    def __call__(self, t):
        print("Warning - undefined")
        return None

    def T(self, p):
        """
        Calculates Transmissivity
        
        :param p:   solution vector
        :param Q:   flow rate m^3/s
        :return T:  transmissivity m^2/s
        """
        return 0.1832339 * self.Q / p[0]

    def S(self, p):
        """
        Calculates Storativity
        
        :param p:   solution vector
        :param Q:   flow rate m^3/s
        :return S:  storativity -
        """
        return 2.2458394 * self.T(p) * p[1] / self.r ** 2

    def trial(self, p, df):  # loglog included: derivatives are missing at the moment.
        """
        Display data and calculated solution together
        
        The function trial allows to produce a graph that superposes data
        and a model. This can be used to test graphically the quality of a
        fit, or to adjust manually the parameters of a model until a
        satisfactory fit is obtained.
        
        :param p:   solution vector
        :param df:  pandas dataframe with two named vectores [t,s]
        """
        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(df.t, self.__call__(p, df.t), df.t, df.s, 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(df.t, self.__call__(p, df.t), df.t, df.s, 'o')
        ax2.set_ylabel('s')
        ax2.set_xlabel('t')
        ax2.grid()
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.show()
        print('T = ', self.T(p), 'm2/s')
        print('S = ', self.S(p), '-')
        print('Ri = ', self.RadiusOfInfluence(p, df.t), 'm')

    def fit(self, p, df, option='lm', output='all'):
        """
        Fit the model parameter of a given model.
        
        The function optimizes the value of the parameters of the model so that
        the model fits the observations. The fit is obtained by an iterative
        non linear least square procedure. This is why the function requires an
        initial guess of the parameters, that will then be iterativly modified
        until a local minimum is obtained.
        
        :param p:       vector of initial guess for the parameters
        :param df:      data set
        :param option:  Levenberg-Marquard (lm is default) or Trust Region Reflection algorithm (trf)
        :param output:  all (default), p, Detailled
            
        :return res.p:  s   detailled optimize results from the scipy.optimize.least_squares 
        :return res_p.x:    solution vector p
        :return tc:         model time 
        :return sc:         model drawdown 
        :return mr:         mean residual between model and data set
        :return sr:         2 standard deviation between model and data set
        :return rms:        root-mean-square between model and data set        
        """
        t = df.t
        s = df.s

        # costfunction
        def fun(p, t, s):
            return np.array(s) - self.__call__(p, t)

        if option == 'lm':
            # Levenberg-Marquard -- Default
            res_p = least_squares(fun, p, args=(t, s), method='lm', xtol=1e-10, verbose=1)
        elif option == 'trf':
            # Trust Region Reflective algorithm
            res_p = least_squares(fun, p, jac='3-point', args=(t, s), method='trf', verbose=1)
        else:
            raise Exception('Specify your option')

        if output == 'all':  # -- Default
            # define regular points to plot the calculated drawdown
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                             dtype=np.float64)
            sc = self.__call__(res_p.x, tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun ** 2))
            return res_p.x, tc, sc, mr, sr, rms
        elif output == 'p':
            return res_p.x
        elif output == 'Detailled':
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                             dtype=np.float64)
            sc = self.__call__(res_p.x, tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun ** 2))
            return res_p, tc, sc, mr, sr, rms
        else:
            raise Exception('The output needs to specified: p or all')

        # Derived daughter classes

class theis(AnalyticalInterferenceModels):
    """ 
    The Theis (1935) solution assumes that the aquifer is confined,
    homogeneous, isotropic, and infinite. The well as radius that is
    negligible. It is pumped at a constant rate Q and is 100 percent
    efficient.   
 
    Under these assumptions, Theis solution can be expressed as: 
 
        s(r,t) = Q/(4 pi T) E1( r2S / 4Tt)
 
    where Q is the pumping rate, T the transmissivity, r the radial 
    distance between the pumping well and the observation well, 
    S the storativity coefficient and t the time.  

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    
    :Example:
    >>> q = 1.3888e-2 #pumping rate in m3/s
    >>> d = 250 #radial distance in m
    >>> theis_model=ht.theis(Q=q,r=d)
    >>> theis_model.plot_typecurve()
    >>> p0 = theis_model.guess_params(data)
    >>> p  = theis_model.fit(p0, data)
    >>> theis_model.trial(p,data)
    """
    def dimensionless(self, td):
        """
        Calculates the dimensionless drawdown for a given dimensionless reduced time td/rd^2
        
        :param td:  dimensionless time
        :return sd: dimensionless drawdown
        """
        return 0.5 * E1(1, 0.25 / td)

    def dimensionless_logderivative(self, td):
        """
        Calculates the dimensionless drawdown derivative for a given dimensionless reduced time td/rd^2
        
        :param td:  dimensionless time
        :return sd: dimensionless derivative
        """
        return 0.5 * np.exp(-0.25 / td)

    def dimensionless_laplace(self, pd):
        """
        Drawdown of the Theis Function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        return 1 / pd * mp.besselk(0, mp.sqrt(pd))

    def dimensionless_laplace_derivative(self, pd):
        """
        Derivative of the Theis Function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown_derivative(td, option='Stehfest')
        """
        return 0.5 * mp.besselk(1, mp.sqrt(pd)) / mp.sqrt(pd)

    def __call__(self, p, t):
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(p, sd)
        return s

    def guess_params(self, df):
        """
        First guess for the parameters of the Theis model.
        
        :param df.t: time
        :param df.s: drawdown
        
        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0 
        """
        n = len(df) / 3
        return get_logline(df[df.index > n])

    def RadiusOfInfluence(self, p, t):
        """
        Calculates the radius of influence
        
        :param p:   solution vector
        :param t:   dimensional time
        :return ri: radius of influence m
        """
        return 2 * np.sqrt(self.T(p) * t[len(t) - 1] / self.S(p))

    def plot_typecurve(self):
        """
        Draw a series of typecurves of Theis (1935).
        """
        td = np.logspace(-1, 4)
        sd = self.dimensionless(td)
        dd = self.dimensionless_logderivative(td)

        plt.loglog(td, sd, td, dd, '--')
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-1, 1e4))
        plt.ylim((1e-2, 10))
        plt.grid('True')
        plt.legend(['Theis', 'Derivative'])
        plt.show()


class theis_noflow(AnalyticalInterferenceModels):
    """
    Theis model with a no-flow boundary.
    
    This is a Theis model in a confined aquifer with an impermeable boundary.
       
    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    
    :Example:
    >>> q = 0.0132 #pumping rate in m3/s
    >>> d = 20 #radial distance in m
    >>> t_noflow=ht.theis_noflow(Q=q,r=d)
    >>> t_noflow.plot_typecurve()
    >>> p0 = t_noflow.guess_params(data)
    >>> p  = t_noflow.fit(p0, data)
    >>> t_noflow.trial(p,data)
    """
    def dimensionless(self, td, Rd):
        """
        Dimensionless drawdown of the Theis model with a no-flow boundary
        
        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """
        ths = theis()
        return ths.dimensionless(td) + ths.dimensionless(td / Rd ** 2)

    def dimensionless_logderivative(self, td, Rd):
        """
        Dimensionless drawdown derivative of the Theis model with a no-flow boundary
        
        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return dd: dimensionless drawdown derivative
        """
        ths = theis()
        return ths.dimensionless_logderivative(td) + ths.dimensionless_logderivative(td / Rd ** 2)

    def dimensionless_laplace(self, pd):
        """
        Drawdown of the Theis with no-flow boundary function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        return 1 / pd * mp.besselk(0, mp.sqrt(pd)) + 1 / (pd) * mp.besselk(0, mp.sqrt(pd) * self.Rd)

    def dimensionless_laplace_derivative(self, pd):
        """
        Drawdown derivative of the Theis with no-flow boundary function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown_derivative(td, option='Stehfest')
        """
        return 0.5 * mp.besselk(1, mp.sqrt(pd)) / mp.sqrt(pd) + 0.5 * mp.besselk(1, mp.sqrt(pd) * self.Rd) / mp.sqrt(
            pd) * self.Rd

    def __call__(self, p, t):
        Rd = np.sqrt(p[2] / p[1])
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless(td, Rd)
        s = self._dimensional_drawdown(p, sd)
        return s

    def guess_params(self, df):
        """
        First guess for the parameters of the Theis model with a no-flow boundary
        
        :param df.t: time
        :param df.s: drawdown
        
        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0 
        :return p[2]: time of intersection between the 2 straight lines
        """
        n = len(df) / 4
        p_late = get_logline(df[df.index > n])
        p_early = get_logline(df[df.index < 2 * n])
        return np.array([p_late[0] / 2, p_early[1], p_late[1] ** 2 / p_early[1]])

    def RadiusOfInfluence(self, p, t):
        """
        Calculates the radius of influence
        
        :param p:   solution vector
        :param t:   dimensional time
        :return ri: radius of influence m
        """
        return np.sqrt(2.2458394 * self.T(p) * p[2] / self.S(p))

    def plot_typecurve(self, Rd=np.array([1.3, 3.3, 10, 33])):
        """
        Type curves of the Theis model with a no-flow boundary
        """
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            sd = self.dimensionless(td, Rd[i])
            dd = self.dimensionless_logderivative(td, Rd[i])
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=Rd[i])
            plt.loglog(td, dd, '-.', color=color)
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-2, 20))
        plt.grid('True')
        plt.legend()
        plt.show()


# class theis_superposition(AnalyticalModels):

class theis_constanthead(AnalyticalInterferenceModels):
    """
    Theis (1941) model with a constant head boundary.
    
    Computes the drawdown at time t for a constant rate pumping test in 
    a homogeneous confined aquifer bounded by a constant head boundary.
    
    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    
    :Example:
    >>> q = 0.03 #pumping rate in m3/s
    >>> d = 20 #radial distance in m
    >>> t_head=ht.theis_constanthead(Q=q,r=d)
    >>> t_head.plot_typecurve()
    >>> p0 = t_head.guess_params(data)
    >>> p  = t_head.fit(p0, data)
    >>> t_head.trial(p,data)
    
    :Reference: Theis, C.V., 1941. The effect of a well on the flow of a 
    nearby stream. Transactions of the American Geophysical Union, 22(3):
    734-738.
    """
    def dimensionless(self, td, Rd):
        """
        Dimensionless drawdown of the Theis model with constant head boundary
        
        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """
        ths = theis()
        return ths.dimensionless(td) - ths.dimensionless(td / Rd ** 2)

    def dimensionless_logderivative(self, td, Rd):
        """
        Dimensionless drawdown derivative of the Theis model with constant head boundary
        
        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """        
        ths = theis()
        return ths.dimensionless_logderivative(td) - ths.dimensionless_logderivative(td / Rd ** 2)

    def dimensionless_laplace(self, pd):
        """
        Drawdown of the Theis with constant head boundary function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        return 1 / pd * mp.besselk(0, mp.sqrt(pd)) - 1 / (pd) * mp.besselk(0, mp.sqrt(pd) * self.Rd)

    def dimensionless_laplace_derivative(self, pd):
        """
        Drawdown derivative of the Theis with constant head boundary function in Laplace domain
        
        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """        
        return 0.5 * mp.besselk(1, mp.sqrt(pd)) / mp.sqrt(pd) - 0.5 * mp.besselk(1, mp.sqrt(pd) * self.Rd) / mp.sqrt(
            pd) * self.Rd

    def __call__(self, p, t):
        Rd = np.sqrt(p[2] / p[1])
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless(td, Rd)
        s = self._dimensional_drawdown(p, sd)
        return s

    def guess_params(self, df):
        """
        First guess for the parameters of the Theis model with a constant head boundary
        
        :param df.t: time
        :param df.s: drawdown
        
        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0 
        :return p[2]: time of intersection between the 2 straight lines
        """
        n = len(df) / 4
        p_late = get_logline(df[df.index > n])
        p_early = get_logline(df[df.index < 2 * n])
        return np.array([p_early[0], p_early[1], 2 * p_late[1] * p_early[1] ** 2 / p_late[0] ** 2])

    def RadiusOfInfluence(self, p):
        """
        Calculates the radius of influence
        
        :param p:   solution vector
        :return ri: radius of influence m
        """
        return np.sqrt(2.2458394 * self.T(p) * p[2] / self.S(p))

    def plot_typecurve(self, Rd=np.array([1.5, 3, 10, 30])):
        """
        Type curves of the Theis model with a constant head boundary
        """
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            sd = self.dimensionless(td, Rd[i])
            dd = self.dimensionless_logderivative(td, Rd[i])
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=Rd[i])
            plt.loglog(td, dd, '-.', color=color)
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-2, 20))
        plt.grid('True')
        plt.legend()
        plt.show()


# Parent generic class

class AnalyticalSlugModels():
    def __init__(self, Q=None, r=None, rw=None, rc=None, cD=None):
        self.Q = Q
        self.r = r
        self.rw = rw
        self.rc = rc
        self.rD = r/rc
        self.cD = cD

    def _dimensionless_time(self, p, t):
        return 0.445268 * t /  p[1] * self.rD ** 2

    def _dimensional_drawdown(self, p, sd):
        return 0.868589 * p[0] * np.float64(sd)

    def _laplace_drawdown(self, td, option='Stehfest'):  # default stehfest
        return list(map(lambda x: mp.invertlaplace(self.dimensionless_laplace, x, method=option, dps=10, degree=16), td))

    def _laplace_drawdown_derivative(self, td, option='Stehfest'):  # default stehfest
        return list(map(
            lambda x: mp.invertlaplace(self.dimensionless_laplace_derivative, x, method=option, dps=10, degree=16), td))

    def __call__(self, t):
        print("Warning - undefined")
        return None
    
    def T(self, p):
        return 0.1832339 * self.Q / p[0]

    def S(self, p):
        return 2.2458394 * self.T(p) * p[1] / self.r ** 2
    
    def Cd(self, p):
        if self.cD is not None:
            return self.cD
        else:
            return self.rc ** 2 / 2 / self.rw ** 2 / self.S(p)

    def trial(self, p, df):
        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(df.t, list(self.__call__(p, df.t)), df.t, df.s, 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(df.t, list(self.__call__(p, df.t)), df.t, df.s, 'o')
        ax2.set_ylabel('s')
        ax2.set_xlabel('t')
        ax2.grid()
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.show()
        print('T = ', self.T(p), 'm2/s')
        print('S = ', self.S(p), '-')
        print('Cd = ', self.Cd(p), '-')

    def fit(self, p, df, option='lm', output='all'):
        t = df.t
        s = df.s

        # costfunction
        def fun(p, t, s):
            return np.array(s) - self.__call__(p, t)

        if option == 'lm':
            # Levenberg-Marquard -- Default
            res_p = least_squares(fun, p, args=(t, s), method='lm', xtol=1e-10, verbose=1)
        elif option == 'trf':
            # Trust Region Reflective algorithm
            res_p = least_squares(fun, p, jac='3-point', args=(t, s), method='trf', verbose=1)
        else:
            raise Exception('Specify your option')

        if output == 'all':  # -- Default
            # define regular points to plot the calculated drawdown
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                             dtype=np.float64)
            sc = self.__call__(res_p.x, tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun ** 2))
            return res_p.x, tc, sc, mr, sr, rms
        elif output == 'p':
            return res_p.x
        elif output == 'Detailled':
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                             dtype=np.float64)
            sc = self.__call__(res_p.x, tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun ** 2))
            return res_p, tc, sc, mr, sr, rms
        else:
            raise Exception('The output needs to specified: p or all')


# Derived daughter classes

class CooperBredehoeftPapadopulos(AnalyticalSlugModels):
    """
    CooperBredehoeftPapadopulos
    -----------------------------
    
    :param Q:   pumping rate
    :param r:   distance between the observation well and pumping well
    :param rw:  radius if the well
    :param rc:  radius of the casing
    
    
    """

    def dimensionless_laplace(self, pd):
        sp = mp.sqrt(pd)
        return mp.besselk(0, self.rD * sp) / (pd * (sp * mp.besselk(1, sp) + self.cd * pd * mp.besselk(0, sp)))

    def dimensionless_laplace_derivative(self, pd):
        sp = mp.sqrt(pd)
        cds= self.cd*sp
        k0 = mp.besselk(0,sp)
        k1 = mp.besselk(1,sp)
        kr0 = mp.besselk(0,sp*self.rD)
        kr1 = mp.besselk(1,sp*self.rD)
        return 0.5*((2*self.cd-1)*kr0*k0+kr1*k1+cds*kr1*k0-cds*kr0*k1)/(mp.power(sp*k1+self.cd*pd*k0, 2))

    def __call__(self, p, t):
        self.cd = self.Cd(p)
        td = self._dimensionless_time(p, t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(p, sd)
        return s

    def guess_params(self, df):
        n = 3*len(df) / 4
        return get_logline(df[df.index > n])

    def plot_typecurve(self, cD=10**np.array([1, 2, 3, 4, 5]), rD = 1):
        self.rD = rD
        td = np.logspace(-1, 3)
        plt.figure()
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            sd = list(self._laplace_drawdown(td*cD[i]))
            dd = list(self._laplace_drawdown_derivative(td*cD[i]))
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=cD[i])
            plt.loglog(td, dd, '-.', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-1, 1e3))
        plt.ylim((1e-1, 1e1))
        plt.grid('True')
        plt.legend()
        plt.show()   
        td = np.logspace(-2, 8)
        plt.figure()
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            sd = list(self._laplace_drawdown(td))
            dd = list(self._laplace_drawdown_derivative(td))
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=cD[i])
            plt.loglog(td, dd, '-.', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-2, 1e8))
        plt.ylim((1e-2, 1e2))
        plt.grid('True')
        plt.legend()
        plt.show()           


class special(AnalyticalInterferenceModels):  # ?? used ?? Theis

    def calc_sl_du(self, Rd):
        sldu = []
        for i in range(0, np.size(Rd)):
            if Rd[i] <= 1:
                Rd[i] = 1.00001
            sldu.append(np.log(Rd[i] ** 2) / ((Rd[i] ** 2 - 1) * Rd[i] ** (-2 * Rd[i] ** 2 / (Rd[i] ** 2 - 1))))
        return sldu

    def calc_inverse_sl_du(self, fri):
        if fri < 2.71828182850322:
            print('Problem in the inversion of Rd: calc_sl_du')
            Rd = 1.000001
        else:
            Rd = np.exp(fri / 2)
            if Rd < 50:
                y = np.linspace(1.00001, 60, 2000)
                x = self.calc_sl_du(y)
                frd = interpolate.interp1d(x, y)
                Rd = frd(fri)
        return Rd
