#    Copyright (C) 2020 by
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
from scipy.special import gamma, gammaincc
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
import mpmath as mp
import openhytest as ht
import pandas as pda


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
    return q / (2 * np.pi * (s1 - s2)) * np.log(r2 / r1)


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


def moy_std(q, s0, l, rw):
    """
    The Moye (1967) formula is used to estimate the transmissivity during a
    pseudo-steady state injection test. It is based on the Dupuit formula
    for confined homogeneous aquifer and steady flow. In addition it
    assumes that the radius of influence of the test is equal to half of
    the length of the test section.

    The method is known to overestimate the transmissivity. Errors of more
    than one order of magnitude are possible.
    :param q: flow rate, m^3/s
    :param s0: drawdown, m
    :param l: length of the section, m
    :param rw: radius of the well, m
    :return T: transmissivity, m^2/s

    :References: Moye, D.G. (1967) Diamond drilling for foundation
    exploration. Civil En. Trans., Inst. Eng. Australia. Apr. 1967, pp.  95-100
    """
    return q * (1 + np.log(l * rw / 2)) / (2 * np.pi * s0)

def get_logline(self, df):
    logt = np.log10(df.t).values
    Gt = np.array([logt, np.ones(logt.shape)])
    p = np.linalg.inv(Gt.dot(Gt.T)).dot(Gt).dot(df.s)
    p[1] = 10 ** (-p[1] / p[0])
    self.p = p
    return self.p

def log_plot(self):
    fig = plt.figure()
    fig.set_size_inches(self.xsize, self.ysize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time in seconds')
    ax1.set_ylabel('Drawdown in meters')
    ax1.set_title(self.ttle)
    ax1.loglog(self.df.t, self.df.s, c='r', marker='+', linestyle='', label='Drawdown')
    ax1.loglog(self.der.t, self.der.s, c='b', marker='x', linestyle='', label='Derivative')
    ax1.loglog(self.tc, self.sc, c='g', label=self.model_label)
    ax1.loglog(self.derc.t, self.derc.s, c='y', label='Model derivative')
    ax1.grid(True)
    ax1.legend()
    return fig

# Parent generic class
class AnalyticalInterferenceModels():
    def __init__(self, Q=None, r=None, Rd=None, p=None, df=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df

    def _dimensionless_time(self, t):
        """
        Calculates dimensionless time
        """
        return (t / (0.5628 * self.p[1])) * 0.25

    def _dimensional_drawdown(self, sd):
        """
        Calculates the dimensional drawdown
        """
        return (sd * self.p[0] * 2) / 2.302585092994046

    def _dimensional_flowrate(self, qd):
        """
        Calculates the dimensional flow rate
        """
        return (np.float64(qd) * np.log(10)) / 2.0 / self.p[0]

    def _stehfest(self, td, degree=12):
        """
        Numerical Laplace inversion with the Stefhest method

        :param x:       vector of the parameters of the function
        :param td:      dimensionless time
        :param degree:  parameter of the Stefhest algorithm (default: 12)
        :return sd:     dimensionless drawdown

        :Reference:
        Widder, D. (1941). The Laplace Transform. Princeton.
        Stehfest, H. (1970). Algorithm 368: numerical inversion of Laplace transforms. Communications of the ACM 13(1):47-49, http://dx.doi.org/10.1145/361953.361969

        """
        if self.degree is None:
            self.degree = degree
        elif degree != 12:
            self.degree = degree
        if self.degree % 2 != 0:
            print('ERROR: The degree of the stehfest algorithm needs to be even!')

        # Calculates stehfest weighting coefficents
        if (self.stehfest_param_inv is None) or (np.size(self.stehfest_param_inv)!=self.degree):
            self.stehfest_param_inv = np.zeros(self.degree)
            for i in range(1,self.degree+1):
                vi_dummy = 0
                for k in range(int((i+1)/2), np.amin([i, self.degree/2])):
                    vi_dummy = vi_dummy + k ** (self.degree/2)*np.prod(range(1,2*k))/(np.prod(range(1,self.degree/2-k)) * np.prod(range(1,k)) * np.prod(range(1,k-1)) * np.prod(range(1,i-k)) * np.prod(range(1,2*k-i)))
                self.stehfest_param_inv[i-1] = (-1) ** (self.degree/2 + i) * vi_dummy

        # Gaver-stehfest algorithm
        sd = np.zeros(np.size(td))
        for i in range(1, self.degree+1):
            pd = i * np.log(2)/td
            sd[i-1] = np.log(2) / td * np.sum(self.stehfest_param_inv[i-1] * eval(self.dimensionless_laplace(pd)) )
        return sd

    def _laplace_drawdown(self, td, option='Stehfest', degrees=12):  # default stehfest
        """
        Alternative calculation with Laplace inversion

        :param td:      dimensionless time
        :param x:       dummy parameter for inversion
        :param option:  Stehfest (default, dps=10, degree=16), dehoog
        :return sd:     dimensionless drawdown
        """
        s = map(lambda x: mp.invertlaplace(self.dimensionless_laplace, x, method=option, dps=10, degree=degrees), td)
        return list(s)

    def _laplace_drawdown_derivative(self, td, option='Stehfest', degrees=12):  # default stehfest
        """
        Alternative calculation with Laplace inversion

        :param td:      dimensionless time
        :param x:       dummy parameter for inversion
        :param option:  Stehfest (default, dps=10, degree=16), dehoog
        :return dd:     dimensionless drawdown derivative
        """
        d = list(map(
            lambda x: mp.invertlaplace(self.dimensionless_laplace_derivative, x, method=option, dps=10, degree=degrees), td))
        return d

    def __call__(self, t):
        print("Warning - undefined")
        return None

    def T(self):
        """
        Calculates Transmissivity

        :return Transmissivity:  transmissivity m^2/s
        """
        return np.log(10)/np.pi/4 * self.Q / self.p[0]

    def S(self):
        """
        Calculates Storativity

        :return Storativity:  storativity -
        """
        return 2.2458394 * self.T() * self.p[1] / self.r ** 2

    def trial(self, p=np.nan):  # loglog included: derivatives are missing at the moment.
        """
        Display data and calculated solution together

        The function trial allows to produce a graph that superposes data
        and a model. This can be used to test graphically the quality of a
        fit, or to adjust manually the parameters of a model until a
        satisfactory fit is obtained.

        :param p:   a solution vector can be initialized
        """
        if np.isnan(p):
            p = self.p

        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(self.df.t, self.__call__(self.df.t), self.df.t, self.df.s, 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(self.df.t, self.__call__(self.df.t), self.df.t, self.df.s, 'o')
        ax2.set_ylabel('s')
        ax2.set_xlabel('t')
        ax2.grid()
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.show()
        print('T = ', self.T(), 'm2/s')
        print('S = ', self.S(), '-')
        print('Ri = ', self.RadiusOfInfluence(), 'm')

    def fit(self, option='lm', fitmethod=None, fitbnds=None):
        """
        Fit the model parameter of a given model.

        The function optimizes the value of the parameters of the model so that
        the model fits the observations. The fit is obtained by an iterative
        non linear least square procedure. This is why the function requires an
        initial guess of the parameters, that will then be iterativly modified
        until a local minimum is obtained.

        :param option:  Levenberg-Marquard (lm is default), Trust Region Reflection algorithm (trf)
        using least-squares implementation from scipy-optimize or minimize function from the same library
        :return res_p.x:    solution vector p
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod
        if fitbnds is not None:
            self.fitbnds = fitbnds
        if self.p is None:
            print("Error, intialize p using the function guess_params")
        
        if option is None:
            print('No fit is performed and given p is returned')
            return self.p       

        t = self.df.t
        s = self.df.s
        p = self.p

        # costfunction
        def fun(p, t, s):
            self.p = p
            return s.to_numpy() - self.__call__(t.to_numpy())

        if option == 'lm':
            # Levenberg-Marquard -- Default
            res_p = least_squares(fun, p, args=(t, s), method='lm', xtol=1e-10, verbose=1)
        elif option == 'trf':
            # Trust Region Reflective algorithm
            res_p = least_squares(fun, p, jac='3-point', args=(t, s), method='trf', verbose=1)
        elif option == 'min':
            # minimize function from scipy.optimize.minimize
            res_p = minimize(fun, p, args=(t, s), method=self.fitmethod, jac=None, bounds=self.fitbnds) #CHECK !!!! NOT WORKING
        else:
            raise Exception('Specify your option')

        # define regular points to plot the calculated drawdown
        self.tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                              dtype=np.float64)
        self.sc = self.__call__(self.tc)
        test = ht.preprocessing(df=pda.DataFrame(data={"t": self.tc, "s": self.sc}))
        self.derc = test.ldiffs()
        self.mr = np.mean(res_p.fun)
        self.sr = 2 * np.nanstd(res_p.fun)
        self.rms = np.sqrt(np.mean(res_p.fun ** 2))
        self.p = res_p.x
        self.detailled_p = res_p
        return res_p.x

# Derived daughter classes

class Theis(AnalyticalInterferenceModels):
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

    :Initialization:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated drawdown
    :param derc: Calculated drawdown derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :paramRadInfluence: Radius of influence m
    :param detailled_p: detailled solution struct from the fit function

    :Example:
    >>> q = 1.3888e-2 #pumping rate in m3/s
    >>> d = 250 #radial distance in m
    >>> theis_model=ht.theis(Q=q,r=d, df=data)
    >>> theis_model.plot_typecurve()
    >>> theis_model.guess_params()
    >>> theis_model.fit()
    >>> theis_model.trial()
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

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(sd)
        return s

    def __init__(self, Q=None, r=None, df=None, p=None):
        self.Q = Q
        self.r = r
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None

    def guess_params(self):
        """
        First guess for the parameters of the Theis model.

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        """
        n = len(self.df) / 3
        self.p = get_logline(self, df=self.df[self.df.index > n])
        return self.p

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return RadInfluence: radius of influence m
        """
        return 2 * np.sqrt(self.T() * self.df.t[len(self.df.t) - 1] / self.S())

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

    def rpt(self, option_fit='lm', ttle='Theis (1935)', author='Author', filetype='pdf', reptext='Report_ths'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Theis (1935) model'

        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Radius of Investigation Ri : {:0.4g} m'.format(self.RadInfluence), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')

class Theis_noflow(AnalyticalInterferenceModels):
    """
    Theis model with a no-flow boundary.

    This is a Theis model in a confined aquifer with an impermeable boundary.

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated drawdown
    :param derc: Calculated drawdown derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :param RadInfluence: Distance to the image well m
    :param detailled_p: detailled solution struct from the fit function

    :Example:
    >>> q = 0.0132 #pumping rate in m3/s
    >>> d = 20 #radial distance in m
    >>> t_noflow=ht.theis_noflow(Q=q,r=d, df=data)
    >>> t_noflow.plot_typecurve()
    >>> t_noflow.guess_params()
    >>> t_noflow.fit()
    >>> t_noflow.trial()
    """

    def __init__(self, Q=None, r=None, Rd=None, df=None, p=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None

    def dimensionless(self, td):
        """
        Dimensionless drawdown of the Theis model with a no-flow boundary

        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """
        ths = Theis()
        return ths.dimensionless(td) + ths.dimensionless(td / self.Rd ** 2)

    def dimensionless_logderivative(self, td):
        """
        Dimensionless drawdown derivative of the Theis model with a no-flow boundary

        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return dd: dimensionless drawdown derivative
        """
        ths = Theis()
        return ths.dimensionless_logderivative(td) + ths.dimensionless_logderivative(td / self.Rd ** 2)

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

    def __call__(self, t):
        self.Rd = np.sqrt(self.p[2] / self.p[1])
        td = self._dimensionless_time(t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(sd)
        return s

    def guess_params(self):
        """
        First guess for the parameters of the Theis model with a no-flow boundary

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        :return p[2]: time of intersection between the 2 straight lines
        """
        n = len(self.df) / 4
        p_late = get_logline(self, self.df[self.df.index > n])
        p_early = get_logline(self, self.df[self.df.index < 2 * n])
        self.p = np.array([p_late[0] / 2, p_early[1], p_late[1] ** 2 / p_early[1]])
        return self.p

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return ri: radius to image well m
        """
        return np.sqrt(2.2458394 * self.T() * self.p[2] / self.S())

    def plot_typecurve(self, Rd=np.array([1.3, 3.3, 10, 33])):
        """
        Type curves of the Theis model with a no-flow boundary
        """
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            self.Rd = Rd[i]
            sd = self.dimensionless(td)
            dd = self.dimensionless_logderivative(td)
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

    def rpt(self, option_fit='lm', ttle='Theis (1935) no flow', author='Author', filetype='pdf', reptext='Report_thn'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Theis (1935) model with no-flow boundary'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Distance distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Radius to image well Rd : {:0.4g} m'.format(self.RadInfluence), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept ti : {:0.2g} m'.format(self.p[2]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')


# class theis_superposition(AnalyticalModels):

class Theis_constanthead(AnalyticalInterferenceModels):
    """
    Theis (1941) model with a constant head boundary.

    Computes the drawdown at time t for a constant rate pumping test in
    a homogeneous confined aquifer bounded by a constant head boundary.

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated drawdown
    :param derc: Calculated drawdown derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :paramRadInfluence: Distance to the image well m
    :param detailled_p: detailled solution struct from the fit function

    :Example:
    >>> q = 0.03 #pumping rate in m3/s
    >>> d = 20 #radial distance in m
    >>> t_head=ht.theis_constanthead(Q=q,r=d, df=data)
    >>> t_head.plot_typecurve()
    >>> t_head.guess_params()
    >>> t_head.fit()
    >>> t_head.trial()

    :Reference: Theis, C.V., 1941. The effect of a well on the flow of a
    nearby stream. Transactions of the American Geophysical Union, 22(3):
    734-738.
    """

    def __init__(self, Q=None, r=None, Rd=None, df=None, p=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None

    def dimensionless(self, td):
        """
        Dimensionless drawdown of the Theis model with constant head boundary

        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """
        ths = Theis()
        return ths.dimensionless(td) - ths.dimensionless(td / self.Rd ** 2)

    def dimensionless_logderivative(self, td):
        """
        Dimensionless drawdown derivative of the Theis model with constant head boundary

        :param td:  dimensionless time
        :param Rd:  dimensionless radial distance
        :return sd: dimensionless drawdown
        """
        ths = Theis()
        return ths.dimensionless_logderivative(td) - ths.dimensionless_logderivative(td / self.Rd ** 2)

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

    def __call__(self, t):
        self.Rd = np.sqrt(self.p[2] / self.p[1])
        td = self._dimensionless_time(t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(sd)
        return s

    def guess_params(self):
        """
        First guess for the parameters of the Theis model with a constant head boundary

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        :return p[2]: time of intersection between the 2 straight lines
        """
        n = len(self.df) / 4
        p_late = get_logline(self, self.df[self.df.index > n])
        p_early = get_logline(self, self.df[self.df.index < 2 * n])
        self.p = np.array([p_early[0], p_early[1], 2 * p_late[1] * p_early[1] ** 2 / p_late[0] ** 2])
        return self.p

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return ri: Distance to image well m
        """
        return np.sqrt(2.2458394 * self.T() * self.p[2] / self.S())

    def plot_typecurve(self, Rd=np.array([1.5, 3, 10, 30])):
        """
        Type curves of the Theis model with a constant head boundary
        """
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            self.Rd = Rd[i]
            sd = self.dimensionless(td)
            dd = self.dimensionless_logderivative(td)
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

    def rpt(self, option_fit='lm', ttle='Theis (1935) const. head', author='Author', filetype='pdf',
            reptext='Report_thc'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Theis (1935) model with const. head boundary'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Distance to image well Rd : {:0.4g} m'.format(self.RadInfluence), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept ti : {:0.2g} m'.format(self.p[2]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')

class JacobLohman(AnalyticalInterferenceModels):
    """
    Constant head test:
    Jacob & Lohman (1952) discharge solution in the well

    :Initialzation:
    :param s: drawdown, m
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.q (m^3/s) for test time respective drawdown, df.s is calculated as 1 / df.q
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param qc: Calculated flow rate
    :param derc: Calculated flow rate derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :paramRadInfluence: Radius of influence m
    :param detailled_p: detailled solution struct from the fit function

    :Example:

    """
    def __init__(self, s=None, r=None, Rd=None, df=None, p=None):
        self.s = s
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None

    def dimensionless_laplace(self, pd):
        """
        Dimensionless flow rate of the Jacob-Lohamn model in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_flowrate(td, option='Stehfest')
        """
        return mp.besselk(1, mp.sqrt(pd)) / (mp.sqrt(pd) * mp.besselk(0, mp.sqrt(pd)))

    def dimensionless_laplace_derivative(self, pd):
        """
        Dimensionless flow rate derivative of the Jacob-Lohamn model in Laplace domain
        """
        return None

    def __call__(self, t):
        td = self._dimensionless_time(t)
        qd = self._laplace_drawdown(td, degrees=12)
        q = self._dimensional_flowrate(qd)
        return q

    def guess_params(self):
        """
        First guess for the parameters of the Jacob & Lohman (1952) model

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        """
        self.df['s'] = 1 / self.df.q
        n = len(self.df) / 3
        self.p = get_logline(self, self.df[self.df.index > n])
        self.df['s'] = self.df.q
        return self.p

    def T(self):
        """
        Calculates Transmissivity

        :return Transmissivity:  transmissivity m^2/s
        """
        return 0.1832339 / self.s / self.p[0]

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return ri: Distance to image well m
        """
        return 2*np.sqrt(self.T() * self.df['t'].iloc[-1] / self.S())

    def perrochet(self, td):
        """
        Perrochet approximation for Jacob & Lohman (1952)

        :return qd:
        """
        return 1 / np.log(1 + np.sqrt(np.pi * td))

    def plot_typecurve(self):
        """
        Type curves of the Jacob-Lohman (1952) model
        """
        plt.figure(1)
        td = np.logspace(-4, 10)
        g = 0.57721566
        ax = plt.gca()
        qd = self._laplace_drawdown(td, degrees=16)
        sd = list(map(lambda x: 1/x, qd))
        d = {'td': td, 'sd': sd}
        df = pda.DataFrame(data=d)
        test = ht.preprocessing(data=df, npoints=50)
        der = test.ldiffs()
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, qd, '-', color=color, label='q_D')
        plt.loglog(der.td, der.sd, '-.', color=color, label='der. 1/q_D')
        plt.xlabel('$t_D$')
        plt.ylabel('$q_D$')
        plt.xlim((1e-3, 1e10))
        plt.ylim((1e-2, 1e2))
        plt.grid('True')
        plt.legend()
        plt.show()
        plt.figure(2)
        ax = plt.gca()
        q1 = 0.5 + 1 / np.sqrt(np.pi * td)
        q2 = 2 / E1(1, 0.25 / td)
        q3 = 0.5 + 1 / np.sqrt(np.pi * td) - 0.25 * np.sqrt(td / np.pi) + td / 8
        q4 = 2 / (np.log(4 * td) - 2 * g) - 2 * g / (np.log(4 * td) - 2 * g) ** 2
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, qd, '-', color=color, label='Jacob-Lohman')
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, q1, '-.', color=color, label='Jacob early asymptote')
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, q2, '-.', color=color, label='Jacob late asymptote')
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, q3, '--', color=color, label='Carslaw early asymptote')
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, q4, '--', color=color, label='Carslaw late asymptote')
        color = next(ax._get_lines.prop_cycler)['color']
        plt.loglog(td, self.perrochet(td), '+', color=color, label='Perrochet approx.')
        plt.xlabel('$t_D$')
        plt.ylabel('$q_D$')
        plt.xlim((1e-3, 1e10))
        plt.ylim((1e-2, 1e2))
        plt.grid('True')
        plt.title('Asymptotes of the Jacob and Lohman (1952) solution')
        plt.legend()
        plt.show()
        residual = qd - self.perrochet(td)
        mr = np.mean(residual)
        print('Mean residual between Jacob-Lohman and Perrochet approx. ', mr, 'm^3/s')
        sr = 2 * np.std(residual)
        print('Standard deviation ', sr, 'm^3/s')
        rms = np.sqrt(np.mean(residual ** 2))
        print('Root-mean-square ', rms, 'm^3/s')

    def rpt(self, option_fit='lm', ttle='Jacob & Lohman (1952)', author='Author', filetype='pdf',
            reptext='Report_jlq'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Jacob & Lohman model'

        self.ttle = ttle
        fig = plt.figure()
        fig.set_size_inches(self.xsize, self.ysize)
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Time in seconds')
        ax1.set_ylabel('Flow rate m^3/s')
        ax1.set_title(self.ttle)
        ax1.loglog(self.df.t, self.df.s, c='r', marker='+', linestyle='', label='q')
        ax1.loglog(self.tc, self.sc, c='g', label=self.model_label)
        ax1.grid(True)
        ax1.legend()

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Hydraulic head : {:3.2e} m'.format(self.s), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Radius of influence Rd : {:0.4g} m'.format(self.RadInfluence), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')

class Warren_Root(AnalyticalInterferenceModels):
    """
    Warren_Root (1936) model for confined double porosity aquifer.

    When the density of fracture is high, but when the porous matrix plays a 
    significant role in the storage capacity of the aquifer, the aquifer
    behaviour can be modeled with the help of the double porosity model. This
    model consider that the flow is occurring mainly in the fracture while the
    water is mainly stored in the porous matrix.

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param Rd:  dimensionless radial distance
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated drawdown
    :param derc: Calculated drawdown derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :paramRadInfluence: Distance to the image well m
    :param detailled_p: detailled solution struct from the fit function
    :param landa: inter-porosity flow parameters
    :param sigma: ratio between the matrix and fracture storativity

    :Reference: Warren, J. E., and P. J. Root (1963), The behaviour of naturally 
    fractured reservoirs, Society of Petroleum Engineers Journal, 3, 245-255.
    """

    def __init__(self, Q=None, r=None, Rd=None, df=None, p=None, sigma=None, landa=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None
        self.sigma = sigma
        self.landa = landa


    def dimensionless_laplace(self, pd):
        """
        Drawdown of the Warren & Root in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        return 1 / pd * mp.besselk(0, mp.sqrt(pd + (self.landa * self.sigma * pd)/(self.sigma * pd + self.landa)))

    def dimensionless_laplace_derivative(self, pd):
        """
        Drawdown derivative of the Warren & Root in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        return None

    def __call__(self, t):
        td = self._dimensionless_time(t)
        self.landa = 2.2458394 * self.p[1] * mp.log(self.p[2]/self.p[1]) / self.p[3]
        self.sigma = (self.p[2]-self.p[1]) / self.p[1]
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(np.float64(sd))
        return s

    def guess_params(self):
        """
        First guess for the parameters of the Theis model with a constant head boundary

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for the early time asymptote
        :return p[2]: intercept with the horizontal axis for the late time asymptote
        :return p[3]: time of the minimum of the derivative
        """
        n = len(self.df) / 4
        p_late = get_logline(self, self.df[self.df.index > n])
        p_early = get_logline(self, self.df[self.df.index < 2 * n])
        warren = ht.preprocessing(df=self.df)
        warren.ldiffs()
        tt = warren.der.t.to_numpy()
        self.p = np.array([p_late[0], p_early[1], p_late[1], tt[np.argmin(warren.der.s.to_numpy())]])
        return self.p

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return ri: Distance to image well m
        """
        return np.sqrt(2.2458394 * self.T() * self.p[2] / self.S2())

    def S2(self):
        """
        Calculates the storativity for the matrix

        :return : storativity of matrix
        """
        return 2.2458394 * self.T() * self.p[2] / self.r ** 2 - self.S()

    def plot_typecurve(self, landa=0.1, sigma=[10, 100, 1000]):
        """
        Different type curves of the Warren & Root model
        """
        self.landa=landa
        td = np.logspace(-2, 7)
        fig, ax = plt.subplots(1,1)
        for i in range(0, len(sigma)):
            self.sigma = sigma[i]
            sd = list(self._laplace_drawdown(td))
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(td, sd, '-', color=color, label= '$\sigma$ = {}'.format(sigma[i]))
            ax.loglog(dummy.der.t, dummy.der.s, ':', color=color)
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.title('$\lambda$ = {}'.format(landa))
        plt.legend()
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-3, 10))
        plt.grid('True')
        plt.legend()
        plt.show()

        landa = [1, 0.1, 0.01]
        self.sigma = 100
        fig, ax = plt.subplots(1,1)
        for i in range(0, len(landa)):
            self.landa= landa[i]
            sd = list(self._laplace_drawdown(td))
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(td, sd, '-', color=color, label= '$\lambda$ = {}'.format(landa[i]))
            ax.loglog(dummy.der.t, dummy.der.s, ':', color=color)
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.title('$\sigma$ = {}'.format(self.sigma))
        plt.legend()
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-3, 10))
        plt.grid('True')
        plt.legend()
        plt.show()

    def rpt(self, option_fit='lm', ttle='Theis (1935) const. head', author='Author', filetype='pdf',
            reptext='Report_thc'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativityf = self.S()
        self.Storativitym = self.S2()
        self.RadInfluence = self.RadiusOfInfluence()
        self.landa = 2.2458394 * self.p[1] * mp.log(self.p[2]/self.p[1]) / self.p[3]
        self.sigma = (self.p[2]-self.p[1]) / self.p[1]
        self.model_label = 'Warren  & Root (1963)'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity Tf : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity Sf : {:3.2e} '.format(self.Storativityf), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Storativity Sm : {:3.2e} '.format(self.Storativitym), fontsize=14,
                 transform=plt.gcf().transFigure)    
        fig.text(1.05, 0.45, 'Inter-porosity flow lambda: {:3.2e} '.format(self.landa), fontsize=14,
                 transform=plt.gcf().transFigure)      
        #fig.text(1.05, 0.45, 'Distance to image well Rd : {:0.4g} m'.format(self.RadInfluence), fontsize=14,
        #         transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'minimum derivative tm : {:0.2g} m'.format(self.p[3]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')       

class GRF(AnalyticalInterferenceModels):
    """
    Barker (1988) general radial flow model

    This class computes the dimensionless drawdown as a function of
    dimensionless time for a constant rate intereference test
    with the General Radial Flow model of Barker (1988). This solution is
    a generalisation of flow equation in 1D, 2D, 3D and non integer flow
    dimension.

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param rw: radius of the well, m
    :param rD: dimensionless radius
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param self:
    :param p: solution vector
    :param der:  Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated drawdown
    :param derc: Calculated drawdown derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize:  size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :paramRadInfluence: Radius of influence m
    :param detailled_p: detailled solution struct from the fit function

    :Reference:
    Barker, J.A. 1988. A Generalized radial flow model fro hydraulic tests
    in fractured rock. Water Resources Research 24, no. 10: 1796-1804.

    """
    def _dimensionless_time(self, t):
        """
        Calculates dimensionless time for GRF
        """
        return (t / (2.2458 * self.p[1]))

    def _dimensional_drawdown(self, sd):
        """
        Calculates the dimensional drawdown for GRF
        """
        return np.float64(sd) * self.p[0] * 0.868588963806504


    def dimensionless(self, td):
        """
        Dimensionless drawdown of the GRF

        :param td:  dimensionless time
        :return sd: dimensionless drawdown
        """
        n = self.p[2]
        u = self.rD ** 2 / (4 * td )
        return self.rD ** (2-n) / (4 * np.pi**(n/2)) * gamma(n/2 - 1)*gammaincc(n/2 - 1,u)

    def dimensionless_laplace(self, pd):
        """
        Drawdown of the General Radial Flow model in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, option='Stehfest')
        """
        n = self.p[2]
        return self.rD**(2-n) * (self.rD**2 * pd/4)**(n/4-0.5) * mp.besselk(n/2-1, self.rD*mp.sqrt(pd)) / pd /mp.gamma(n/2)

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s

    def __init__(self, Q=None, r=1, rw=1, df=None, p=None):
        self.Q = Q
        self.r = r
        self.rw = rw
        self.rD = r / rw
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.RadInfluence = None
        self.detailled_p = None
        self.degree = None
        self.stehfest_param_inv = None
        self.fitmethod = None
        self.fitbnds = None

    def guess_params(self):
        """
        First guess for the parameters of the GRF model.

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        :return p[2]: flow dimension, default radial
        """
        n = len(self.df) / 3
        p = get_logline(self, df=self.df[self.df.index > n])
        self.p = [p[0], p[1]*(self.rw/self.r)**2, 2.0]
        return self.p

    def plot_typecurve(self, rD = None):
        """
        Draw a series of typecurves of the General Flow Model.
        """
        if rD is None:
            rD = 1
        if self.rD is None:
            self.rD = rD
        td = np.logspace(-1, 6) * self.rD ** 2
        plt.figure(1)
        ax = plt.gca()
        for n in np.linspace(1, 3, 9):
            self.p = np.array([0,0,n])
            color = next(ax._get_lines.prop_cycler)['color']
            sd = list(self._laplace_drawdown(td))
            ax.loglog(td, sd, '-', color=color, label=n)
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-1, 1e+6))
        plt.grid('True')
        plt.legend()
        plt.show()

        plt.figure(2,  figsize=(10,10))
        for i in range(0,4):
            plt.subplot(2,2,i+1)
            self.p = np.array([0,0,1.5+i*0.5])
            sd = list(self._laplace_drawdown(td))
            #dd = list(self._laplace_drawdown_derivative(td))
            d = {'td': td, 'sd': sd}
            df = pda.DataFrame(data=d)
            test = ht.preprocessing(df=df, npoints=20)
            der = test.ldiff()
            plt.loglog(td, sd, '-', der.td, der.sd, '.-')
            plt.title('r_D=%g, n=%g' % (self.rD ,1.5+i*.5))
            plt.xlabel('$t_D$')
            plt.ylabel('$s_D$')
            plt.ylim((1e-2, 1e+2))
            plt.grid('True')
        plt.show()

    def rpt(self, option_fit='lm', ttle='GRF', author='Author', filetype='pdf',
            reptext='Report_grf'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.model_label = 'GRF model'

        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Well radius : {:0.4g} m '.format(self.rw), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.7, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.60, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Flow dimension n : {:3.2e} '.format(self.p[2]), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Equivalent transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.45, 'Equivalent storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.35, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')

class StorativityInterferenceModels(AnalyticalInterferenceModels):
    def __init__(self):
        pass

    def _dimensionless_time(self, t):
        return 0.445268 * t / self.p[1] * self.rD ** 2

    def _dimensional_drawdown(self, sd):
        return 0.868589 * self.p[0] * np.float64(sd)

    def _laplace_drawdown(self, td, option='Stehfest'):  # default stehfest
        return list(
            map(lambda x: mp.invertlaplace(self.dimensionless_laplace, x, method=option, dps=10, degree=12), td))

    def _laplace_drawdown_derivative(self, td, option='Stehfest'):  # default stehfest
        return list(map(
            lambda x: mp.invertlaplace(self.dimensionless_laplace_derivative, x, method=option, dps=10, degree=12), td))

    def __call__(self, t):
        print("Warning - undefined")
        return None

    def T(self):
        return 0.1832339 * self.Q / self.p[0]

    def S(self):
        return 2.2458394 * self.T() * self.p[1] / self.r ** 2

    def S2(self):
        if self.cD is not None:
            return self.rc ** 2 / 2 / self.rw ** 2 / self.cD
        else:
            return None

    def Cd(self):
        if self.cD is not None:
            return self.cD
        else:
            return self.rc ** 2 / 2 / self.rw ** 2 / self.S()

    def trial(self):
        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(self.df.t, list(self.__call__(self.df.t)), self.df.t, self.df.s, 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(self.df.t, list(self.__call__(self.df.t)), self.df.t, self.df.s, 'o')
        ax2.set_ylabel('s')
        ax2.set_xlabel('t')
        ax2.grid()
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.show()
        print('T = ', self.T(), 'm2/s')
        print('S = ', self.S(), '-')
        print('Cd = ', self.Cd(), '-')


# Derived daughter classes

class PapadopulosCooper(StorativityInterferenceModels):
    """
    Interference test with the Papadopulos & Cooper (1967) solution

    :param Q: pumping rate
    :param r: distance between the observation well and pumping well
    :param rw: radius of the well
    :param rc: radius of the casing
    :param rD: dimensionless radius
    :param cD: dimensionless well bore storage coefficient
    :param p: solution vector
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param der: Drawdown derivative from the input data given as dataframe with der.t and der.s
    :param tc: Calculated time
    :param sc: Calculated draw down
    :param derc: Calculated flow rate derivative data given as dataframe with derc.t and derc.s
    :param mr: mean resiuduals from the fit function
    :param sr: standard derivative from the fit function
    :param rms: root-mean-square from the fit function
    :param ttle: title of the plot
    :param model_label: model label of the plot
    :param xsize: size of the plot in x (default is 8 inch)
    :param ysize: size of the plot in y (default is 6 inch)
    :param Transmissivity: Transmissivity m^2/s
    :param Storativity: Storativtiy -
    :param Storativity2: Storativtiy -
    :param RadInfluence: Radius of influence m
    :param detailled_p: detailled solution struct from the fit function

    :Description:
    The Papadopulos-Cooper (1967) solution for a constant rate pumping test in a large diameter well.
    The aquifer is confined and homogeneous.

    It is assumed that there is no skin effect. However the storativity coefficient and the wellbore storage coefficient are assumed independent.
    Two different possibilities exist to solve for storativity. It is equivalent to say that the storativity near the well is different from those of the aquifer.

    The dimensionless wellbore storage coefficient is: Cd = rc^2/(2*rw^2*S)

    Note that in the original publication of Cooper et al.
    The dimensionless parameter was alpha, it is related to Cd by: alpha = 1 / (2 Cd)

    :Reference:
    Papadopulos, I.S., and H.H.J. Cooper. 1967. Drawdown in a
    well of large diameter. Water Resources Research 3, no. 1: 241-244.

    """
    def __init__(self, Q=None, r=None, rw=None, rc=None, cD=None, df=None, p=None):
        self.Q = Q
        self.r = r
        self.rw = rw
        self.rc = rc
        self.rD = r / rc
        self.cD = cD
        self.p = p
        self.df = df
        self.der = None
        self.tc = None
        self.sc = None
        self.derc = None
        self.mr = None
        self.sr = None
        self.rms = None
        self.ttle = None
        self.model_label = None
        self.xsize = 8
        self.ysize = 6
        self.Transmissivity = None
        self.Storativity = None
        self.Storativity2 = None
        self.RadInfluence = None
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None

    def dimensionless_laplace(self, pd):
        sp = mp.sqrt(pd)
        return mp.besselk(0, self.rD * sp) / (pd * (sp * mp.besselk(1, sp) + self.cd * pd * mp.besselk(0, sp)))

    def dimensionless_laplace_derivative(self, pd):
        sp = mp.sqrt(pd)
        cds = self.cd * sp
        k0 = mp.besselk(0, sp)
        k1 = mp.besselk(1, sp)
        kr0 = mp.besselk(0, sp * self.rD)
        kr1 = mp.besselk(1, sp * self.rD)
        return 0.5 * ((2 * self.cd - 1) * kr0 * k0 + kr1 * k1 + cds * kr1 * k0 - cds * kr0 * k1) / (
            mp.power(sp * k1 + self.cd * pd * k0, 2))

    def __call__(self, t):
        self.cd = self.Cd()
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s

    def guess_params(self):
        n = 3 * len(self.df) / 4
        self.p = get_logline(self, self.df[self.df.index > n])
        return self.p

    def plot_typecurve(self, cD=10 ** np.array([1, 2, 3, 4, 5]), rD=1):
        """
        Type curves of the Cooper-Bredehoeft-Papadopulos (1967) model
        """
        self.rD = rD
        td = np.logspace(-1, 3)
        plt.figure(1)
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            sd = list(self._laplace_drawdown(td * cD[i]))
            dd = list(self._laplace_drawdown_derivative(td * cD[i]))
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
        plt.figure(2)
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

    def rpt(self, option_fit='lm', ttle='Papadopulos-Cooper (1967)', author='Author', filetype='pdf',
            reptext='Report_cbp'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        self.fit(option=option_fit)

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.Storativity2 = self.S2()

        self.model_label = 'Cooper-Bredehoeft-Papadopulos (1967)'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Well radius : {:0.4g} m '.format(self.rw), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.7, 'Casing radius: {:0.4g} m '.format(self.rc), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.65, 'Distance to pumping well : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.6, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Aquifer Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        if self.cD is not None:
            fig.text(1.05, 0.45, 'Well surroundings storativity S2 : {:3.2e} '.format(self.Storativity2), fontsize=14,
                     transform=plt.gcf().transFigure)
        self.cD = self.Cd()
        fig.text(1.05, 0.4, 'Wellbore storage : {:0.4g} '.format(self.cD), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1, 0.35, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
