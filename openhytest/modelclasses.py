#    Copyright (C) 2021 by
#    Nathan Dutler <nathan.dutler@exquiro.ch>
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
   Copyright (C) 2021 openhytest Developers
   Nathan Dutler <nathan.dutlern@unine.ch>
   Philippe Renard <philippe.renard@unine.ch>
   Bernard Brixel <bernard.brixel@erdw.ethz.ch>

"""

import numpy as np
from scipy.special import expn as E1
from scipy.special import gamma, gammaincc, factorial, kv
from scipy.special import airy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, Bounds
from scipy.interpolate import interp1d
import mpmath as mp
import pandas as pda
import openhytest as ht

# Utilities

def thiem(q, s1, s2, r1, r2):
    """
    Calculate the transmissivity with Thiem (1906) solution

    The Thiem method requires to know the pumping rate and the drawdown in
    two observation wells located at different distances from the pumping
    well. It assumes steady-state conditions.

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
    ax1.loglog(self.der.t, np.abs(self.der.s), c='b', marker='x', linestyle='', label='Derivative')
    ax1.loglog(self.tc, self.sc, c='g', label=self.model_label)
    ax1.loglog(self.derc.t, np.abs(self.derc.s), c='y', label='Model derivative')
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
        self.inversion_M = None
        self.inversion_V = None
        self.inversion_s = None


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
        return qd * np.log(10) / 2.0 / self.p[0]

    def _laplace_drawdown(self, td, inversion_option=None):
        """
        Alternative calculation with Laplace inversion

        :param td:      dimensionless time
        :param inversion_option:  stehfest or dehoog
        :return sd:     dimensionless drawdown
        """
        if inversion_option is not None:
            self.inversion_option = inversion_option

        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace, td)
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace, td)
        return sd


    def _laplace_drawdown_types(self, td, inversion_option=None):
        """
        Alternative calculation with Laplace inversion

        :param td:      dimensionless time
        :param inversion_option:  stehfest or dehoog
        :return sd:     dimensionless drawdown
        """
        if inversion_option is not None:
            self.inversion_option = inversion_option

        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace_types, td)
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace_types, td)
        return sd


    def _laplace_drawdown_derivative(self, td, inversion_option=None):
        """
        Alternative calculation with Laplace inversion

        :param td:      dimensionless time
        :param inversion_option:  stehfest or dehoog
        :return dd:     dimensionless drawdown derivative
        """
        if inversion_option is not None:
            self.inversion_option = inversion_option

        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace_derivative, td)
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace_derivative, td)
        return sd


    def _coeff(self, fitcoeff=16):
        """
        Calculates the coefficent for the stehfest method.

        :param fitcoeff: number of coefficients for inversion (Default is 12)
        :return self.inversion_V: gives the inversion coefficent for stehfest
        :return self.inversion_M: number of coefficients for inversion
        :return V: gives the inversion coefficent for stehfest
        """
        if fitcoeff != 16:
            M = np.int(self.fitcoeff)
        else:
            M = fitcoeff # Default

        if M % 2 > 0: # Check if M is even
            M = M + 1
        V = np.zeros(M)
        for i in range(1,M+1):
            vi = 0
            for k in range(np.int((i+1)/2),np.int(np.min([i,M/2]))+1):
                vi = vi + (k**(M/2)*factorial(2*k))/(factorial(np.int(M/2-k))*factorial(k)*factorial(k-1)*factorial(i-k)*factorial(2*k-i))
            V[i-1] = (-1) ** ((M/2)+i)*vi
        self.inversion_V = V
        self.inversion_M = M
        return V


    def stehfest(self, Fp, td):
        """
        Numerical Laplace inversion with the Stehfest method

        :return self.inversion_s: the calculated drawdown in time domain
        :return s: the calculated drawdown in time domain

        :References: Widder, D. (1941). The Laplace Transform. Princeton.
        Stehfest, H. (1970). Algorithm 368: numerical inversion of Laplace transforms.
        Communications of the ACM 13(1):47-49, http://dx.doi.org/10.1145/361953.361969
        """
        p = np.zeros([self.inversion_M, np.size(td)])
        for i in range(1,self.inversion_M+1):
            p[i-1] = i*np.log(2)/td
        uu = Fp(p)
        VV = np.repeat(self.inversion_V, np.size(td)).reshape(self.inversion_M, np.size(td))
        su = np.multiply(VV, uu)
        s = np.log(2)/td*sum(su)
        self.inversion_s = s
        return s


    def dehoog(self, Fp, td, alpha=0, tol=1e-9, M=20):
        """
        Numerical Laplace inversion with the dehoog method

        de Hoog et al's quotient difference method with accelerated convergence for the continued fraction expansion

        Modification: The time vector td is split in segments of equal magnitude
        which are inverted individually. This gives a better overall accuracy.

        :return self.inversion_s: the calculated drawdown in time domain
        :return s: the calculated drawdown in time domain

        :References: de Hoog, F. R., Knight, J. H., and Stokes, A. N. (1982). An improved
        method for numerical inversion of Laplace transforms. S.I.A.M. J. Sci.
        and Stat. Comput., 3, 357-366.

        with modifications after Hollenbeck, K. J. (1998) INVLAP.M: A matlab function for numerical
        inversion of Laplace transforms by the de Hoog algorithm, http://www.isva.dtu.dk/staff/karl/invlap.htm
        """
        if self.fitcoeff is not None:
            M = self.fitcoeff

        logallt = np.log10(td)
        iminlogall = np.int(np.floor(np.nanmin(logallt)))
        imaxlogall = np.int(np.ceil(np.nanmax(logallt)))
        iminlogallt = np.int(iminlogall)
        imaxlogallt = np.int(imaxlogall)
        f = []
        for ilogt in range(iminlogallt, imaxlogallt+1):
            t = td[((logallt>=ilogt) & (logallt<ilogt+1))]
            if len(t) > 0:
                T = 2 * np.max(t)
                gamma = alpha - np.log(tol) / (2*T)
                run = np.linspace(0,2*M, 2*M+1)
                p = gamma + 1j * np.pi * run / T
                a = Fp(p)      # evaluate function
                a[0] = a[0] / 2 # zero term is halved
                # build up e and q tables.
                e = np.zeros((2*M+1, M+1), dtype=complex)
                q = np.zeros((2*M, M+1), dtype=complex)
                q[:,1] = a[1:2*M+1]/a[0:2*M]
                for r in np.arange(1,M+1):
                    e[0:2*(M-r)+1,r] = q[1:2*(M-r)+2,r] - q[0:2*(M-r)+1,r] + e[1:2*(M-r)+2,r-1]
                    if r < M:
                        rq = r + 1
                        q[0:2*(M-rq)+2,rq] = q[1:2*(M-rq)+3,rq-1]*e[1:2*(M-rq)+3,rq-1]/e[0:2*(M-rq)+2,rq-1]
                # build up d vector
                d = np.zeros((2*M+1,1), dtype=complex)
                d[0] = a[0]
                d[1:2*M:2] = np.vstack(-q[0,1:M+1])
                d[2:2*M+1:2] = np.vstack(-e[0,1:M+1])
                # build up A and B matrix
                A = np.zeros((2*M+2,len(t)), dtype=complex)
                B = np.zeros((2*M+2,len(t)), dtype=complex)
                A[1,:] = d[0,0]*np.ones((1,len(t)))
                B[0,:] = np.ones((1,len(t)))
                B[1,:] = np.ones((1,len(t)))
                z = np.exp(1j*np.pi*t/T)
                for n in np.arange(2, 2*M+2):
                    A[n,:] = A[n-1,:] + d[n-1]*np.ones((1,len(t)))*z*A[n-2,:]
                    B[n,:] = B[n-1,:] + d[n-1]*np.ones((1,len(t)))*z*B[n-2,:]

                h2M = .5 * ( np.ones((1,len(t))) + ( d[2*M-1]-d[2*M] ) * np.ones((1,len(t))) * z )
                R2Mz = -h2M*(np.ones((1,len(t))) - (np.ones((1,len(t))) + d[2*M]*np.ones((1,len(t))) * z / h2M ** 2) ** 5)
                A[2*M+1,:] = A[2*M,:] + R2Mz * A[2*M-1,:]
                B[2*M+1,:] = B[2*M,:] + R2Mz * B[2*M-1,:]
                fpiece = np.array(1/T * np.exp(gamma * t) * np.real(A[2*M+1,:] / B[2*M+1,:]))
                f = np.append(f, np.hstack(fpiece))
        self.inversion_s = f
        return f


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


    def trial(self, p=None, inversion_option=None):  # loglog included: derivatives are missing at the moment.
        """
        Display data and calculated solution together

        The function trial allows to produce a graph that superposes data
        and a model. This can be used to test graphically the quality of a
        fit, or to adjust manually the parameters of a model until a
        satisfactory fit is obtained.

        :param p:   a solution vector can be initialized
        """

        if inversion_option is not None:
            self.inversion_option = inversion_option
        else:
            if self.inversion_option is None:
                self.inversion_option = 'dehoog'

        if p is not None:
            p = self.p

        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(self.df.t.to_numpy(), self.__call__(self.df.t.to_numpy()), self.df.t.to_numpy(), self.df.s.to_numpy(), 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(self.df.t.to_numpy(), self.__call__(self.df.t.to_numpy()), self.df.t.to_numpy(), self.df.s.to_numpy(), 'o')
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


    def fit(self, fitmethod=None, fitcoeff=None):
        """
        Fit the model parameter of a given model.

        The function optimizes the value of the parameters of the model so that
        the model fits the observations. The fit is obtained by an iterative
        non linear least square procedure. This is why the function requires an
        initial guess of the parameters, that will then be iterativly modified
        until a local minimum is obtained.

        :param option:  Levenberg-Marquard (lm is default), Trust Region Reflection algorithm (trf) or
        dogbox using least-squares implementation from scipy-optimize or use nofit to caculate only the
        statistic
        :param fitcoeff: The number of coefficent needs to be defined for Laplace inversions.
        :return res_p.x:    solution vector p
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod

        if fitcoeff is not None:
            self.fitcoeff = fitcoeff
        if self.p is None:
            print("Error, intialize p using the function guess_params")

        t = self.df.t
        s = self.df.s
        p = self.p

        if self.inversion_option is not None:
            if self.inversion_option == 'stehfest':
                if self.fitcoeff is None:
                    print('Please, first specifiy the number of coefficient used for the inversion.')
                else:
                    if fitcoeff is None:
                        self.fitcoeff = 16
                    self._coeff()

        # costfunction
        def fun(p, t, s):
            self.p = p
            return s.to_numpy() - self.__call__(t.to_numpy())

        if self.fitmethod == 'lm':
            # Levenberg-Marquardt algorithm (Default).
            # Doesn’t handle bounds and sparse Jacobians.
            # Usually the most efficient method for small unconstrained problems.
            res_p = least_squares(fun, p, args=(t, s), method=self.fitmethod, xtol=1e-10, verbose=1)
        elif self.fitmethod == 'trf':
            # Trust Region Reflective algorithm, particularly suitable for large sparse
            # problems with bounds. Generally robust method.
            res_p = least_squares(fun, p, jac='3-point', args=(t, s), method=self.fitmethod, verbose=1)
            # dogleg algorithm with rectangular trust regions, typical use case is small problems
            # with bounds. Not recommended for problems with rank-deficient Jacobian
        elif self.fitmethod == 'dogbox':
            res_p = least_squares(fun, p, args=(t, s), method=self.fitmethod, verbose=1)

        elif self.fitmethod == 'nofit':
            # Calculates the statistic for a given vector p
            res_p = least_squares(fun, p, args=(t, s), method='trf', max_nfev=1)
        else:
            raise Exception('Choose your fitmethod: lm, trf and dogbox')

        # define regular points to plot the calculated drawdown
        self.tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                              dtype=np.float64)
        self.sc = self.__call__(self.tc)
        test = ht.preprocessing(df=pda.DataFrame(data={"t": self.tc, "s": self.sc}))
        self.derc = test.ldiffs()
        self.mr = np.mean(res_p.fun)
        self.sr = 2 * np.nanstd(res_p.fun)
        self.rms = np.sqrt(np.mean(res_p.fun ** 2))
        self.p = np.float64(res_p.x)
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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

    :Example:
    >>> q = 1.3888e-2 #pumping rate in m3/s
    >>> d = 250 #radial distance in m
    >>> theis_model=ht.theis(Q=q,r=d, df=data)
    >>> theis_model.plot_typecurve()
    >>> theis_model.guess_params()
    >>> theis_model.fit()
    >>> theis_model.trial()

    :Reference: Theis, C.V., 1935. The relation between the lowering of the
    piezometric surface and the rate and duration of discharge of a well using
    groundwater storage, Am. Geophys. Union Trans., vol. 16, pp. 519-524.
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
        :function: _laplace_drawdown(td, inversion_option='stehfest')
        """
        return 1 / pd * kv(0, np.sqrt(pd))

    def dimensionless_laplace2(self, pd): #!!!! can be removed, used to check with stehfest against mpmath laplaceinversion
        """
        Drawdown of the Theis Function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, inversion_option='stehfest')
        """
        return 1 / pd * mp.besselk(0, mp.sqrt(pd))

    def dimensionless_laplace_derivative(self, pd):
        """
        Derivative of the Theis Function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown_derivative(td, inversion_option='stehfest')
        """
        return 0.5 * kv(1, np.sqrt(pd)) / np.sqrt(pd)

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(sd)
        return s

    def __init__(self, Q=None, r=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None

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

    def rpt(self, fitmethod=None, ttle='Theis (1935)', author='Author', filetype='pdf', reptext='Report_ths'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm', 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm' #set Default

        self.fit()

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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

    :Example:
    >>> q = 0.0132 #pumping rate in m3/s
    >>> d = 20 #radial distance in m
    >>> t_noflow=ht.theis_noflow(Q=q,r=d, df=data)
    >>> t_noflow.plot_typecurve()
    >>> t_noflow.guess_params()
    >>> t_noflow.fit()
    >>> t_noflow.trial()
    """

    def __init__(self, Q=None, r=None, Rd=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None

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
        :function: _laplace_drawdown(td, inversion_option='Stehfest')
        """
        return 1 / pd * kv(0, np.sqrt(pd)) + 1 / (pd) * kv(0, np.sqrt(pd) * self.Rd)

    def dimensionless_laplace_derivative(self, pd):
        """
        Drawdown derivative of the Theis with no-flow boundary function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown_derivative(td, inversion_option='Stehfest')
        """
        return 0.5 * kv(1, np.sqrt(pd)) / np.sqrt(pd) + 0.5 * kv(1, np.sqrt(pd) * self.Rd) / np.sqrt(
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

    def rpt(self, fitmethod=None, ttle='Theis (1935) no flow', author='Author', filetype='pdf', reptext='Report_thn'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm' #set Default

        self.fit()

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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

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

    def __init__(self, Q=None, r=None, Rd=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None

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
        :function: _laplace_drawdown(td, inversion_option='Stehfest')
        """
        return 1 / pd * kv(0, np.sqrt(pd)) - 1 / (pd) * kv(0, np.sqrt(pd) * self.Rd)

    def dimensionless_laplace_derivative(self, pd):
        """
        Drawdown derivative of the Theis with constant head boundary function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, inversion_option='Stehfest')
        """
        return 0.5 * kv(1, np.sqrt(pd)) / np.sqrt(pd) - 0.5 * kv(1, np.sqrt(pd) * self.Rd) / np.sqrt(
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

    def rpt(self, fitmethod='lm', ttle='Theis (1935) const. head', author='Author', filetype='pdf',
            reptext='Report_thc'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm' #set Default

        self.fit()

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


class Theis_multirate(AnalyticalInterferenceModels):
    """
    Theis model for multiple rate tests (1935)

    Calculate the drawdown at time t for multiple rate tests without accounting for quadratic head losses.

    The model is designed to interpret drawdown within a piezometer during
    a step drawdown test. It can also be used to interpret simultaneously
    the pumping and recovery data.

    :Initialization:
    :param Q: given in 2 vectors Q.t and Q.q or a scalar as pumping rate, m3/s
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
    :param fitmethod: see fit function for the various options
    :param inversion_option: 'stehfest' or 'dehoog'

    :Reference:

    """

    def dimensionless(self, td):
        """
        Calculates the dimensionless drawdown of the Theis model for multiple rate tests (1935)

        :param td:  dimensionless time
        :return sd: dimensionless drawdown
        """
        s = np.zeros(np.shape(td))
        for i in range(0, np.size(self.Q.t)):
            self.p[0] = self.p[0] * self.diff_Q[i]
            if i == 0:
                s = s + 0.5 * E1(1, 0.25 / td)
            else:
                si = np.array([i for i, x in enumerate(self.df.t.ge(self.Q.t[i-1])) if x], dtype=int)
                s[si] = s[si] + 0.5 * E1(1, 0.25 / (td[si]-self.Q.t[i-1]))
        return s


    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self.dimensionless(td)
        s = self._dimensional_drawdown(sd)
        return s


    def __init__(self, Q=None, r=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None


    def guess_params(self):
        """
        First guess for the parameters of the Theis model for multiple rate tests

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        """
        if np.isscalar(self.Q):
            print('Pumping rate needs to be given by pandas dataframe with Q.t and Q.q vector')
        else:
            self.pumpingrates = np.hstack((self.Q.q[0], np.diff(self.Q.q)))
            self.begintime = np.hstack([0, self.Q.t.iloc[:-1].to_numpy()])

            iss = self.df.t[self.df.t < self.begintime[1]].index

        if np.isscalar(self.Q):
            n = len(self.df) / 3
            self.p = get_logline(self, df=self.df[self.df.index > n])
            self.p[0] = self.p[0] / (self.Q)
        elif self.Q.t is not None:
            test = ht.preprocessing()
            test.birsoy_time(df=self.df, Qmat=self.Q)
            self.p = get_logline(self, df=test.birsoy)

        else:
            print('')


    def rpt(self, fitmethod=None, ttle='Theis (1935)', author='Author', filetype='pdf', reptext='Report_ths'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm', 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        self.fit()

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Theis multirate model'

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


class HantushJacob(AnalyticalInterferenceModels):
    """
    Hantush and Jacob (1955) solution

    Computes the drawdown at time t for a constant rate pumping test in a
    homogeneous, isotropic and leaky confined aquifer of infinite extent.

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
    :param fitmethod: see fit function for the various options
    :param inversion_option: 'stehfest' or 'dehoog'
    :param e: Thickness of the aquitard

    :Reference:


    """

    def dimensionless_laplace(self, pd):
        """
        Hantush-Jacob (1955) Function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, fitmethod='stehfest')
        """
        return 1/pd * kv(0, np.sqrt(pd + self.p[2] ** 2))


    def dimensionless_laplace_derivative(self, pd):
        """
        Derivative of Hantush-Jacob (1955) Function in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown_derivative(td, fitmethod='stehfest')
        """
        return None

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s

    def __init__(self, Q=None, r=None, e=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None
        self.e = e


    def guess_params(self):
        """
        First guess for the parameters of the Hantush and Jacob (1955) solution

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        :return p[2]: r/B
        """
        n = len(self.df) / 3
        x1 = get_logline(self, df=self.df[self.df.index > n])
        ths = ht.Theis(Q=self.Q, r=self.r ,df=self.df, p=x1)
        x1 = ths.fit(fitmethod='lm')
        x2 = np.exp(-self.df.s.iloc[-1]/self.p[0]*2.3/2+0.1)
        if (x2 > 1):
            x2 = np.log(-self.df.s.iloc[-1]/self.p[0]*2.3/2)
        self.p = np.hstack([x1, x2])
        return self.p


    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return RadInfluence: radius of influence m
        """
        return 2 * np.sqrt(self.T() * self.df.t[len(self.df.t) - 1] / self.S())


    def plot_typecurve(self, rb=[2,1,0.3,.1,0.03,0.01]):
        """
        Draw the type curves of Hantush and Jacob (1955)
        """
        td = np.logspace(-2, 5)
        ax = plt.gca()
        self.p = [1,1,1]
        for i in range(0, len(rb)):
            self.p[2] = rb[i]
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
            d = {'td': td, 'sd': sd}
            df = pda.DataFrame(data=d)
            test = ht.preprocessing(df=df, npoints=50)
            der = test.ldiffs()
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=rb[i])
            plt.loglog(der.td, der.sd, '-.', color=color)
        theis = ht.Theis()
        st=theis.dimensionless(td)
        plt.loglog(td, st, 'k--', label='Theis')
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-3, 1e1))
        plt.grid('True')
        plt.legend()
        plt.show()


    def aquitard_conductivity(self):
        """
        Calculates the aquitard conductivity
        """
        B = self.r/self.p[2]
        return self.T() * self.e / B ** 2


    def rpt(self, fitmethod=None, ttle='Hantush&Jacob (1955)', author='openhytest developer', filetype='pdf', reptext='Report_HJ'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm', 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        self.fit()

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RadiusOfInfluence()
        self.model_label = 'Hantush & Jacob (1955) model'
        self.Ka = self.aquitard_conductivity()

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
        fig.text(1.05, 0.70, 'Thickness of aquitard: {:0.4g} m '.format(self.e), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Aquitard conductivity k: {:3.2g} m/s'.format(self.Ka), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'r/B: {:0.2g} '.format(self.p[2]), fontsize=14,
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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

    :Example:

    """
    def __init__(self, s=None, r=None, Rd=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None


    def dimensionless_laplace(self, pd):
        """
        Dimensionless flow rate of the Jacob-Lohamn model in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_flowrate(td, option='Stehfest')
        """
        return kv(1, np.sqrt(pd)) / (np.sqrt(pd) * kv(0, np.sqrt(pd)))

    def dimensionless_laplace_derivative(self, pd):
        """
        Dimensionless flow rate derivative of the Jacob-Lohamn model in Laplace domain
        """
        return None

    def __call__(self, t):
        td = self._dimensionless_time(t)
        qd = self._laplace_drawdown(td)
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
        qd = self._laplace_drawdown(td, inversion_option='dehoog')
        sd = list(map(lambda x: 1/x, qd))
        d = {'td': td, 'sd': sd}
        df = pda.DataFrame(data=d)
        test = ht.preprocessing(df=df, npoints=50)
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

    def rpt(self, fitmethod='lm', ttle='Jacob & Lohman (1952)', author='Author', filetype='pdf',
            reptext='Report_JL'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param fitmethod: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm' #set Default
        self.fit()

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

class WarrenRoot(AnalyticalInterferenceModels):
    """
    WarrenRoot (1936) model for confined double porosity aquifer.

    When the density of fracture is high, but when the porous matrix plays a
    significant role in the storage capacity of the aquifer, the aquifer
    behaviour can be modeled with the help of the double porosity model. This
    model consider that the flow is occurring mainly in the fracture while the
    water is mainly stored in the porous matrix.

    :Initialzation:
    :param Q: pumping rate, m3/s
    :param r: radius between wells, m
    :param rw: radius of the well
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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

    :Reference: Warren, J. E., and P. J. Root (1963), The behaviour of naturally
    fractured reservoirs, Society of Petroleum Engineers Journal, 3, 245-255.
    """

    def __init__(self, Q=None, r=None, rw=None, Rd=None, df=None, p=None, sigma=None, landa=None, inversion_option=None):
        self.Q = Q
        self.r = r
        self.rw = None
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
        self.inversion_option=inversion_option
        self.fitcoeff = None


    def dimensionless_laplace(self, pd):
        """
        Drawdown of the Warren & Root in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown(td, inversion_option='dehoog')
        """
        return 1 / pd * kv(0, np.sqrt(pd + (self.landa() * self.sigma() * pd) / (self.sigma() * pd + self.landa() * np.ones(np.shape(pd)) )))

    def dimensionless_laplace_types(self, pd):
        """
        Drawdown of the Warren & Root in Laplace domain

        :param pd: Laplace parameter
        :function: _laplace_drawdown_types(td, inversion_option='dehoog')
        """
        return 1 / pd * kv(0, np.sqrt(pd + (self.landa * self.sigma * pd)/(self.sigma * pd + self.landa)))

    def landa(self):
        return 2.2458394 * self.p[1] * np.log(self.p[2]/self.p[1]) / self.p[3]

    def sigma(self):
        return (self.p[2]-self.p[1]) / self.p[1]

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s

    def guess_params(self):
        """
        First guess for the parameters of the Theis model with a constant head boundary

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for the early time asymptote
        :return p[2]: intercept with the horizontal axis for the late time asymptote
        :return p[3]: time of the minimum of the derivative
        """
        warren = ht.preprocessing(df=self.df)
        warren.ldiffs()
        tm = warren.der.t.to_numpy()
        dd = np.mean(warren.der.s.iloc[-3:])
        a  = np.log(10)*dd
        t0 = self.df.t.iloc[0] / np.exp(self.df.s.iloc[0]/dd)
        t1 = self.df.t.iloc[-1] / np.exp(self.df.s.iloc[-1]/dd)
        self.p = np.array([a, t0, t1, tm[np.argmin(warren.der.s.to_numpy())]])
        return self.p

    def RadiusOfInfluence(self):
        """
        Calculates the radius of influence

        :return ri: Distance to image well m
        """
        return np.sqrt(2.2458394 * self.T() * self.p[2] / self.S())

    def S2(self):
        """
        Calculates the storativity for the matrix

        :return : storativity of matrix
        """
        return 2.2458394 * self.T() * self.p[2] / self.r ** 2 - self.S()

    def approximation(self, td):
        """
        Calculates the asymptotic approximation

        :return : drawdown
        """
        return 0.5*(np.log(td*4/(1+self.sigma()))-0.5772-E1(1,self.landa()*(1+self.sigma())*td/self.sigma())+E1(1,self.landa()*td/self.sigma()))


    def fit_approximation(self, fitmethod='trf', fitcoeff=16):
        """
        Fit the approximation parameter of the Warren & Root model.

        The function optimizes the value of the parameters of the model so that
        the model fits the observations. The fit is obtained by an iterative
        non linear least square procedure. This is why the function requires an
        initial guess of the parameters, that will then be iterativly modified
        until a local minimum is obtained.

        :param option:  Levenberg-Marquard (lm), Trust Region Reflection algorithm (trf is default) or
        dogbox using least-squares implementation from scipy-optimize or use nofit to caculate only the
        statistic
        :param fitcoeff: The number of coefficent needs to be defined for Laplace inversions.
        :return res_p.x:    solution vector p
        """

        t = self.df.t
        s = self.df.s
        p = self.p

        # costfunction
        def fun(p, t, s):
            self.p = p
            td = self._dimensionless_time(t)
            sd= self.approximation(td)
            sapprox = self._dimensional_drawdown(sd)
            return s.to_numpy() - sapprox

        if fitmethod == 'lm':
            # Levenberg-Marquardt algorithm (Default).
            # Doesn’t handle bounds and sparse Jacobians.
            # Usually the most efficient method for small unconstrained problems.
            res_p = least_squares(fun, p, args=(t, s), method=fitmethod, xtol=1e-10, verbose=1)
        elif fitmethod == 'trf':
            # Trust Region Reflective algorithm, particularly suitable for large sparse
            # problems with bounds. Generally robust method.
            res_p = least_squares(fun, p, jac='3-point', args=(t, s), method=fitmethod, verbose=1)
            # dogleg algorithm with rectangular trust regions, typical use case is small problems
            # with bounds. Not recommended for problems with rank-deficient Jacobian
        elif fitmethod == 'dogbox':
            res_p = least_squares(fun, p, args=(t, s), method=fitmethod, verbose=1)

        elif fitmethod == ' nofit':
            # Calculates the statistic for a given vector p
            res_p = least_squares(fun, p, args=(t, s), method=fitmethod, max_nfev=0)
        else:
            raise Exception('Choose your fitmethod: lm, trf, dogbox or nofit')

        # define regular points to plot the calculated drawdown
        self.tc = np.logspace(np.log10(t[0]), np.log10(t[len(t) - 1]), num=len(t), endpoint=True, base=10.0,
                              dtype=np.float64)
        self.sc = self.__call__(self.tc)
        test = ht.preprocessing(df=pda.DataFrame(data={"t": self.tc, "s": self.sc}))
        self.derc = test.ldiffs()
        self.mr = np.mean(res_p.fun)
        self.sr = 2 * np.nanstd(res_p.fun)
        self.rms = np.sqrt(np.mean(res_p.fun ** 2))
        self.p = np.float64(res_p.x)
        self.detailled_p = res_p
        return res_p.x


    def plot_typecurve(self, landa=0.1, sigma=[10, 100, 1000]):
        """
        Different type curves of the Warren & Root model
        """
        self.landa=landa
        td = np.logspace(-2, 7)
        fig, ax = plt.subplots(1,1)
        for i in range(0, len(sigma)):
            self.sigma = sigma[i]
            sd = self._laplace_drawdown_types(td, inversion_option='dehoog')
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
            sd = self._laplace_drawdown_types(td, inversion_option='dehoog')
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

    def rpt(self, fitmethod='trf', ttle='Warren & Root example', author='openhytest developer', filetype='pdf',
            reptext='Report_wc'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' (default) or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """

        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        Bounds([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        self.fit()
        self.p = np.float64(self.p)
        self.Transmissivity = self.T()
        self.Storativityf = self.S()
        self.Storativitym = self.S2()
        self.RadInfluence = self.RadiusOfInfluence()
        self.landa = 2.2458394 * self.p[1] * np.log(np.float64(self.p[2]/self.p[1])) / self.p[3]
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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

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
        :function: _laplace_drawdown(td, inversion_option='dehoog')
        """
        n = self.p[2]
        return self.rD**(2-n) * (self.rD**2 * pd/4)**(n/4-0.5) * kv(n/2-1, self.rD*np.sqrt(pd)) / pd / gamma(n/2)

    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s

    def __init__(self, Q=None, r=1, rw=1, df=None, p=None, inversion_option=None):
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
        self.fitmethod = None
        self.fitbnds = None
        self.inversion_option=inversion_option
        self.fitcoeff = None

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
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
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
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
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

    def rpt(self, fitmethod='trf', ttle='GRF', author='openhytest developer', filetype='pdf',
            reptext='Report_grf'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        self.fit()

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


class Boulton(AnalyticalInterferenceModels):
    """
    Boulton (1963) solution for unconfined aquifers with delayed yields

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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'
    :param sigma: [sigma, phi]

    :Reference:
    BOULTON, N.S., ANALYSIS OF DATA FROM NON-EQUILIBRIUM PUMPING TESTS ALLOWING FOR DELAYED YIELD FROM STORAGE,
    Proceedings of the Institution of Civil Engineers, 1963, 26:3, 469-482
    """

    def __init__(self, Q=None, r=1, rw=1, sigma=None, df=None, p=None, inversion_option=None):
        self.Q = Q
        self.r = r
        self.rw = rw
        self.sigma = sigma
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
        self.fitmethod = None
        self.fitbnds = None
        self.inversion_option = inversion_option
        self.fitcoeff = None


    def _dimensionless_time(self, t):
        """
        Calculates dimensionless time
        """
        return 0.445268 * t * self.p[1]


    def dimensionless_laplace(self, pd):
        """
        Boulton (1963) Laplace dimensionless domain solution

        :param pd: Laplace parameter
        :param x[0]: sigma = S / S_y
        :param x[1]: phi   = ( alpha r^2 S ) / T
        :function: _laplace_drawdown(td, inversion_option='dehoog')
        """
        if self.sigma is None:
            self.x = np.array([self.p[1]/(self.p[1]+self.p[2]), 2*self.p[3] * self.p[1]])
        else:
            self.x = self.sigma
        return kv(0, np.sqrt(pd + self.x[1] * pd / (self.x[0] * (pd + self.x[1])))) / pd


    def __call__(self, t):
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s


    def guess_params(self):
        """
        First guess for the parameters of the Boulton model.

        :return p[0]: slope of Jacob straight line for late time
        :return p[1]: intercept with the horizontal axis for s = 0
        :return p[2]: intercept with the horizontal axis for s = 0 for the late asymptote
        :return p[3]: phi
        """
        p = np.zeros(4)
        p[1] = self.df.t[0]
        n = len(self.df) / 4
        pj = get_logline(self, df=self.df[self.df.index > n])
        p[0] = pj[0]
        p[2] = pj[1]
        p[3] = 1e-4
        self.p = p
        return self.p


    def plot_typecurve(self, sigma = 0.01):
        """
        Draw a series of typecurves of the Boulton model
        """
        td = np.logspace(-2, 6)
        plt.figure(1)
        ax = plt.gca()
        for i in range(1, 5):
            self.sigma = np.array([0.01, 10**(-i)])
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=self.sigma[1])
        ths = Theis()
        st1 = ths.dimensionless(td)
        plt.loglog(td, st1, ':k', label='Theis')
        st2 = 0.5 * E1(1, (1 + self.sigma[0])/(4*td*self.sigma[0]))
        plt.loglog(td, st2, ':b')
        plt.title('$\sigma$')
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e6))
        plt.ylim((1e-3, 1e2))
        plt.grid('True')
        plt.legend()
        plt.show()

        td = np.logspace(-2, 6)
        plt.figure(2)
        ax = plt.gca()
        for i in range(1, 5):
            self.sigma = np.array([10**(-i), 0.01])
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label=self.sigma[0])
            plt.loglog(dummy.der.t, dummy.der.s, '-.', color=color)
            st2 = 0.5 * E1(1, (1 + self.sigma[0])/(4*td*self.sigma[0]))
            plt.loglog(td, st2, ':', color=color)
        plt.title('$\phi$')
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e6))
        plt.ylim((1e-3, 1e2))
        plt.grid('True')
        plt.legend()
        plt.show()


    def rpt(self, fitmethod='trf', ttle='Boulton', author='openhytest developer', filetype='pdf',
            reptext='Report_blt', p=None):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' (default) or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf'  # set Default

        if p is not None:
            self.p = p
            self.fitmethod = 'nofit'
        else:
            self.inversion_option = 'stehfest'


        self._coeff()
        self.fit()

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.omegad = 2.2458394 * self.Transmissivity * self.p[2] / self.r**2 - self.Storativity
        self.RadInfluence = 2*np.sqrt(self.Transmissivity*self.df.t.iloc[-1] / self.omegad)
        self.model_label = 'Boulton model'

        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14,
                 transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Discharge rate : {:3.2e} m³/s'.format(self.Q), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Radial distance : {:0.4g} m '.format(self.r), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.65, 'Hydraulic parameters :',
                 fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.6, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Drainage porosity : {:3.2e} '.format(self.omegad), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.45, 'Radius of investigation : {:3.2e} '.format(self.RadInfluence), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1, 0.4, 'Fitting parameters :',
                 fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'slope a : {:0.2g} m'.format(
            self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} s'.format(
            self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept t1  : {:0.2g} s'.format(
            self.p[2]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'phi : {:0.2g} m'.format(
            self.p[3]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, 'mean residual : {:0.2g} m'.format(
            self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.05, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.savefig(reptext + '.' + filetype, bbox_inches='tight')


class StorativityInterferenceModels(AnalyticalInterferenceModels):
    def __init__(self):
        pass

    def _dimensionless_time(self, t):
        return 0.445268 * t / self.p[1] * self.rD ** 2

    def _dimensional_drawdown(self, sd):
        return 0.868589 * self.p[0] * np.float64(sd)

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
        ax1.loglog(self.df.t.to_numpy(), self.__call__(
            self.df.t.to_numpy()), self.df.t.to_numpy(), self.df.s.to_numpy(), 'o')
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(self.df.t.to_numpy(), self.__call__(
            self.df.t.to_numpy()), self.df.t.to_numpy(), self.df.s.to_numpy(), 'o')
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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

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
    def __init__(self, Q=None, r=1, rw=None, rc=1, cD=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None

    def dimensionless_laplace(self, pd):
        sp = np.sqrt(pd)
        return kv(0, self.rD * sp) / (pd * (sp * kv(1, sp) + self.cD * pd * kv(0, sp)))

    def dimensionless_laplace_derivative(self, pd):
        sp = np.sqrt(pd)
        cds = self.cD * sp
        k0 = kv(0, sp)
        k1 = kv(1, sp)
        kr0 = kv(0, sp * self.rD)
        kr1 = kv(1, sp * self.rD)
        return 0.5 * ((2 * self.cD - 1) * kr0 * k0 + kr1 * k1 + cds * kr1 * k0 - cds * kr0 * k1) / (
            np.power(sp * k1 + self.cD * pd * k0, 2))

    def __call__(self, t):
        self.cD = self.Cd()
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
            self.cD = cD[i]
            sd = self._laplace_drawdown(td * cD[i], inversion_option='dehoog')
            dd = self._laplace_drawdown_derivative(td * cD[i], inversion_option='dehoog')
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
            self.cD = cD[i]
            sd = self._laplace_drawdown(td, inversion_option='dehoog')
            dd = self._laplace_drawdown_derivative(td, )
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


    def rpt(self, fitmethod='trf', ttle='Papadopulos-Cooper (1967)', author='Author', filetype='pdf',
            reptext='Report_cbp'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        self.fit()

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


class Agarwal(StorativityInterferenceModels):
    """
    Interference test with the Agarwal (1970) solution

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
    :param fitmethod: see fit function for the various options
    :param fitbnds:
    :param inversion_option: 'stehfest' or 'dehoog'

    :Description:
    Computes drawdon in the aquifer for a constant rate pumping test in a large diameter well with skin effect. The aquifer is confined and homogeneous.

    The skin factor is equal to:

    sigma = (T-Ts)/Ts * log( rs / rw )

    The dimensionless well bore storage coefficient is:

    Cd = rc^2/(2 rw^2 S)

    with:
    T  = transmissivity of the aquifer
    Ts = transmissivity of the skin (perturbed zone around the well)
    rs = radius of the skin (from the center of the well)
    rw = radius of the well screen
    S  = storativity of the aquifer

    :Reference:
    Agarwal, RG, R Al-Hussainy and HJ Ramey, 1970,
    An investigation of wellbore storage and skin effect in unsteady liquid flow. SPE Journal: 279-290


    """
    def __init__(self, Q=None, r=1, rw=None, rc=1, cD=None, df=None, p=None, inversion_option=None):
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
        self.inversion_option=inversion_option
        self.fitcoeff = None


    def dimensionless_laplace(self, pd):
        """
        Calculates the dimensionless Agarwala drawdown in the aquifer
        """
        cd = self.p[0]
        rd = self.p[1]
        sg = self.p[2]
        sp = np.sqrt(pd)
        k0 = kv(0, sp)
        k1 = kv(1, sp)
        return kv(0, rd * sp) / (pd * (((1 + pd * cd * sg) * sp * k1) + (cd * pd * k0)))


    def __call__(self, t):
        self.cd = self.Cd()
        td = self._dimensionless_time(t)
        sd = self._laplace_drawdown(td)
        s = self._dimensional_drawdown(sd)
        return s


    def guess_params(self):
        """
        First guess of solution vector using steady-state late time approximation.
        """
        n = 3 * len(self.df) / 4
        p = get_logline(self, self.df[self.df.index > n])
        self.p = [p[0], p[1], 1.0]
        return self.p


    def plot_typecurve(self, rD=1):
        """
        Type curves of the Agarwal (1970) model
        """
        #plt.figure(1, figsize=(10,10))
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
        self.rD = rD
        td = np.logspace(-2, 8)
        for i in range(0, 5):
            for sg in range(0, 15, 5):
                self.p = [10**i, rD, sg]
                sd = self._laplace_drawdown(td, inversion_option='dehoog')
                color = next(ax1._get_lines.prop_cycler)['color']
                ax1.loglog(td, sd, '-', color=color)
        ax1.set_xlabel('$t_D$')
        ax1.set_ylabel('$s_D$')
        ax1.set_xlim((1e0, 1e8))
        ax1.set_ylim((1e-1, 1e2))
        ax1.grid('True')

        td = np.logspace(-1, 4)
        cD = 1e4
        cD2 = 1e3
        cD3 = 5
        pca = PapadopulosCooper(cD = cD, p = [cD, rD])
        sd1 = pca._laplace_drawdown(td*cD, inversion_option='dehoog')
        dd1 = pca._laplace_drawdown_derivative(td*cD, inversion_option='dehoog')
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.loglog(td, sd1, '-', color=color, label='Papadopulos and Cooper model cD = 10000')
        ax2.loglog(td, dd1, '-.', color=color)
        td2 = td * cD2
        self.p = [cD2, rD, (np.log(cD)-np.log(cD2))/2]
        sd2 = self._laplace_drawdown(td2, inversion_option='dehoog')
        d = {'t': td2, 's': sd2}
        df = pda.DataFrame(data=d)
        dummy = ht.preprocessing(df=df)
        dummy.ldiff()
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.loglog(td, sd2, '-', color=color, label='Agarwal model cD = 1000')
        ax2.loglog(dummy.der.t/cD2, dummy.der.s, '-.', color=color)
        self.p = [cD3, rD, (np.log(cD)-np.log(cD3))/2]
        td3 = td * cD3
        sd3 = self._laplace_drawdown(td3, inversion_option='dehoog')
        d2 = {'t': td3, 's': sd3}
        df2 = pda.DataFrame(data=d2)
        dummy2 = ht.preprocessing(df=df2)
        dummy2.ldiff()
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.loglog(td, sd3, '-', color=color, label='Agarwal model cD = 5')
        ax2.loglog(dummy2.der.t/cD3, dummy2.der.s, '-.', color=color)
        ax2.set_xlabel('$t_D / C_D$')
        ax2.set_ylabel('$s_D$')
        ax2.set_title(('rD = 1'))
        ax2.set_xlim((1e-1, 1e4))
        ax2.set_ylim((1e-1, 1e1))
        ax2.grid('True')
        ax2.legend()

        rD = 10
        pca = PapadopulosCooper(cD = cD, p = [cD, rD])
        sd1 = pca._laplace_drawdown(td*cD, inversion_option='dehoog')
        dd1 = pca._laplace_drawdown_derivative(td*cD, inversion_option='dehoog')
        color = next(ax3._get_lines.prop_cycler)['color']
        ax3.loglog(td, sd1, '-', color=color, label='Papadopulos and Cooper model cD = 10000')
        ax3.loglog(td, dd1, '-.', color=color)
        td2 = td * cD2
        self.p = [cD2, rD, (np.log(cD)-np.log(cD2))/2]
        sd2 = self._laplace_drawdown(td2, inversion_option='dehoog')
        d = {'t': td2, 's': sd2}
        df = pda.DataFrame(data=d)
        dummy = ht.preprocessing(df=df)
        dummy.ldiff()
        color = next(ax3._get_lines.prop_cycler)['color']
        ax3.loglog(td, sd2, '-', color=color, label='Agarwal model cD = 1000')
        ax3.loglog(dummy.der.t/cD2, dummy.der.s, '-.', color=color)
        self.p = [cD3, rD, (np.log(cD)-np.log(cD3))/2]
        td3 = td * cD3
        sd3 = self._laplace_drawdown(td3, inversion_option='dehoog')
        d2 = {'t': td3, 's': sd3}
        df2 = pda.DataFrame(data=d2)
        dummy2 = ht.preprocessing(df=df2)
        dummy2.ldiff()
        color = next(ax3._get_lines.prop_cycler)['color']
        ax3.loglog(td, sd3, '-', color=color, label='Agarwal model cD = 5')
        ax3.loglog(dummy2.der.t/cD3, dummy2.der.s, '-.', color=color)
        ax3.set_xlabel('$t_D / C_D$')
        ax3.set_ylabel('$s_D$')
        ax3.set_title(('rD = 10'))
        ax3.set_xlim((1e-1, 1e4))
        ax3.set_ylim((1e-1, 1e1))
        ax3.grid('True')
        ax3.legend()

        rD = 100
        pca = PapadopulosCooper(cD = cD, p = [cD, rD])
        sd1 = pca._laplace_drawdown(td*cD, inversion_option='dehoog')
        dd1 = pca._laplace_drawdown_derivative(td*cD, inversion_option='dehoog')
        color = next(ax4._get_lines.prop_cycler)['color']
        ax4.loglog(td, sd1, '-', color=color, label='Papadopulos and Cooper model cD = 10000')
        ax4.loglog(td, dd1, '-.', color=color)
        td2 = td * cD2
        self.p = [cD2, rD, (np.log(cD)-np.log(cD2))/2]
        sd2 = self._laplace_drawdown(td2, inversion_option='dehoog')
        d = {'t': td2, 's': sd2}
        df = pda.DataFrame(data=d)
        dummy = ht.preprocessing(df=df)
        dummy.ldiff()
        color = next(ax4._get_lines.prop_cycler)['color']
        ax4.loglog(td, sd2, '-', color=color, label='Agarwal model cD = 1000')
        ax4.loglog(dummy.der.t/cD2, dummy.der.s, '-.', color=color)
        self.p = [cD3, rD, (np.log(cD)-np.log(cD3))/2]
        td3 = td * cD3
        sd3 = self._laplace_drawdown(td3, inversion_option='dehoog')
        d2 = {'t': td3, 's': sd3}
        df2 = pda.DataFrame(data=d2)
        dummy2 = ht.preprocessing(df=df2)
        dummy2.ldiff()
        color = next(ax4._get_lines.prop_cycler)['color']
        ax4.loglog(td, sd3, '-', color=color, label='Agarwal model cD = 5')
        ax4.loglog(dummy2.der.t/cD3, dummy2.der.s, '-.', color=color)
        ax4.set_xlabel('$t_D / C_D$')
        ax4.set_ylabel('$s_D$')
        ax4.set_title(('rD = 100'))
        ax4.set_xlim((1e-1, 1e4))
        ax4.set_ylim((1e-1, 1e1))
        ax4.grid('True')
        ax4.legend()


    def rpt(self, fitmethod='trf', ttle='Agarwal (1970)', author='openhytest developer', filetype='pdf',
            reptext='Report_aga'):
        """
        Calculates the solution and reports graphically the results of the pumping test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'trf' #set Default

        self.fit()

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.wbstorage = self.Cd()

        self.model_label = 'Agarwal (1970)'
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
        fig.text(1.05, 0.5, 'Storativity S : {:3.2e} '.format(self.Storativity), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.45, 'Wellbore storage C_D: {:3.2e} '.format(self.wbstorage), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.4, 'Skin factor s_g: {:0.4g} '.format(self.p[2]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1, 0.35, 'Fitting parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'slope a : {:0.2g} m'.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'intercept t0 : {:0.2g} m'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, 'mean residual : {:0.2g} m'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, '2 standard deviation : {:0.2g} m'.format(self.sr), fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.1, 'Root-mean-square : {:0.2g} m'.format(self.rms), fontsize=14,
                 transform=plt.gcf().transFigure)


class Slugtests(AnalyticalInterferenceModels):
    def __init__(self):
        pass

    def _dimensionless_time(self, t):
        return t / self.p[1]*self.p[0]

    def __call__(self, t):
        print("Warning - undefined")
        return None


    def T(self):
        """
        Calculates Transmissivity for slug test

        :return Transmissivity:  transmissivity m^2/s
        """
        Sw = self.Vs * self.Ceff * self.rho * self.g
        return 0.5 * Sw / (self.p[1]*np.pi)

    def S(self):
        """
        Calculates Storativity for slug test

        :return Storativity:  storativity -
        """
        Sw = self.Vs * self.Ceff * self.rho * self.g
        return 0.5 * Sw / (self.p[0] * np.pi * self.rw ** 2)

    def RI_Guyonnet(self):
        """
        Calculates the radius of investigation according to Guyonnet et al. (1993)

        :return radius: radius in meter
        """
        n = 0.462
        m = 0.495
        xlim = 2.5
        tdl = self.p[0] * self.df.t[-1:].to_numpy() / self.p[1]
        x = tdl ** n / self.p[0] ** m
        rdi = 0
        if x < xlim:
            rdi = self.rw * 3.54 * tdl ** n
        else:
            rdi = self.rw * 8.37 * self.p[0] ** m
        return rdi

class Hvorslev(Slugtests):

    """
    Hvorslev (1951) Solution for a slug test using normalized drawdown

    :param rw: radius of the well
    :param rc: radius of the casing
    :param p: solution vector
    :param df: pandas dataframe with two vectors named df.t and normalized df.s for test time respective drawdown
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
    :param RadInfluence: Radius of influence m
    :param detailled_p: detailled solution struct from the fit function


    :Description:
    Solution for a slug test in confined aquifer with negligible storage.

    """

    def __init__(self, rw=None, rc=None, df=None, p=None):
        self.rw = rw
        self.rc = rc
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
        self.fitcoeff = None

    def __call__(self, t):
        return self.dimensionless(t)

    def dimensionless(self, td):
        """
        Dimensionless drawdown Hvosrslev slug model

        :param td:  dimensionless time
        :return sd: dimensionless drawdown
        """
        return np.exp(-td / self.p)


    def guess_params(self):
        """
        First guess for the parameters

        :return p[0]: t0, intercept with the horizontal axis for the early time asymptote
        """
        n = len(self.df) / 4
        p = get_logline(self, df=self.df[self.df.index < n])
        self.p = p[1]
        return self.p


    def plot_typecurve(self, p=1):
        """
        Type curves of the Hvorslev (1951) model
        """
        self.p = p
        td = np.logspace(-1, 3)
        plt.figure(1)
        ax = plt.gca()
        sd = self.dimensionless(td)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.semilogy(td, sd, '-', color=color, label='s')
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((0, 10))
        plt.ylim((1e-5, 1))
        plt.grid('True')
        plt.legend()
        plt.show()


class Neuzil(Slugtests):
    """
    Shut-in pulse or slug test with the Neuzil (1982) solution

    :param rw: radius of the well
    :param Ceff: effective Compressibility of test section
    :param Vs: Volume of the test section
    :param cD: dimensionless well bore storage coefficient
    :param p: solution vector
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param rho: density of water 1000 kg/m^3
    :param g: gravity 9.81 m/s^2
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
    :param fitmethod: see fit function for the various options
    :param inversion_option: 'stehfest' or 'dehoog'

    :Description:
    The aquifer is supposed to be confined, the well fully penetrating and the
    slug injection or withdrawal is supposed to be instantaneous.

    The dimensionless time td is related with well-bore storage, well bore radius, time and transmissivity as followed:
    td / cd = 2 T t / rc^2

    NB: Modified from Cooper et al. (1967) solution for a slug test.
    Note that in the original publication of Cooper et al.
    The dimensionless parameter was alpha, it is related to Cd by: alpha = 1 / (2 Cd)

    :Reference:
    Neuzil, C. E. (1982), On conducting the modified ‘Slug’ test in tight
    formations, Water Resour. Res., 18( 2), 439– 441, doi:10.1029/WR018i002p00439.

    """
    def __init__(self, rw=None, cD=None, df=None, p=None, Ceff=None, Vs=None, inversion_option='stehfest'):
        self.rw = rw
        self.cD = cD
        self.p = p
        self.df = df
        self.Ceff = Ceff
        self.Vs = Vs
        self.der = None
        self.rho = 1000
        self.g = 9.81
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
        self.inversion_option=inversion_option
        self.fitcoeff = None

    def dimensionless_laplace(self, pd):
        """
        Neuzil (1982) Laplace domain solution for slug test (pulse)
        includes dimensionless well-bore storage coefficient

        Original Implementation with alpha cf. P. Hsieh
        """
        sp = np.sqrt(pd)
        return self.cd * kv(0, sp) / (self.cd * pd * kv(0, sp) +  sp * kv(1, sp))

    def __call__(self, t):
        if self.cD is not None:
            self.p[0] = self.Cd
        else:
            self.cd = self.p[0]
        td = self._dimensionless_time(t)
        s = self._laplace_drawdown(td)
        return s


    def guess_params(self):
        """
        Calculates a first set of parameters for the Neuzil (1982) solution

        :return p[0]:  cd
        :return p[1]:  t0
        """
        self.p = np.zeros(2)
        len_df = np.size(self.df.t, 0)
        # Calculating the negative derivative
        dummy = ht.preprocessing(df=self.df)
        dummy.ldiff()
        dummy.ldiffs(npoints = np.uint64(len_df/1.5));
        # finding the (negative) maximum and the corresponding index of the value.
        # (the section with the steepest derivative on a slug test type curve
        # determines the value of alpha (or C_D))
        ind = np.nanargmax(-dummy.der.s)
        dmax = -dummy.der.s[ind]
        # first guess Cd
        #-------------------
        # calculating from analytical solution a type curve relating the maximum of the
        # derivative with Cd; finding from it the corresponding Cd for the found negative
        # maximum of the derivative of s
        dsmax = np.array([0.13906, 0.14019, 0.14307, 0.15001, 0.16413, 0.18562, 0.21037, 0.23401, 0.25427, 0.27064, 0.28361, 0.29381, 0.3019, 0.30843, 0.31381, 0.31823, 0.32202, 0.32525, 0.32802, 0.33047, 0.33265, 0.33452, 0.33628, 0.33779, 0.33919, 0.34048, 0.34164, 0.3427, 0.34367, 0.34457, 0.3454, 0.34619, 0.34693, 0.34761, 0.34825, 0.34884, 0.34937, 0.3499, 0.35043, 0.35091, 0.35132, 0.35176, 0.35218, 0.35253, 0.35291, 0.35327, 0.35354, 0.35392, 0.3542, 0.3545, 0.35478, 0.35502, 0.35531, 0.35551, 0.35579, 0.35597, 0.35624, 0.35641, 0.35665, 0.35682, 0.35703, 0.3572, 0.35737, 0.35756, 0.35768, 0.35789, 0.35801, 0.35819, 0.35833, 0.35844, 0.35862, 0.35872, 0.35887, 0.35901, 0.35907, 0.35925, 0.35936, 0.35943, 0.35959, 0.35968, 0.35976, 0.3599, 0.35999, 0.36005, 0.36019, 0.36027, 0.36031, 0.36046, 0.36054, 0.36058, 0.3607, 0.36079, 0.36085, 0.36091, 0.36102, 0.36109, 0.36112, 0.36122, 0.36131, 0.36136])
        cdmax = np.array([0.01, 0.026561, 0.070548, 0.18738, 0.4977, 1.3219, 3.5112, 9.326, 24.771, 65.793, 174.75, 464.16, 1232.8, 3274.5, 8697.5, 23101, 61359, 162980, 432880, 1149800, 3053900, 8111300, 21544000, 57224000, 151990000, 403700000, 1072300000, 2848000000, 7564600000, 20092000000, 53367000000, 141750000000, 376490000000, 1000000000000, 2656100000000, 7054800000000, 18738000000000, 49770000000000, 132190000000000, 351120000000000, 932600000000000, 2.4771e+15, 6.5793e+15, 1.7475e+16, 4.6416e+16, 1.2328e+17, 3.2745e+17, 8.6975e+17, 2.3101e+18, 6.1359e+18, 1.6298e+19, 4.3288e+19, 1.1498e+20, 3.0539e+20, 8.1113e+20, 2.1544e+21, 5.7224e+21, 1.5199e+22, 4.037e+22, 1.0723e+23, 2.848e+23, 7.5646e+23, 2.0092e+24, 5.3367e+24, 1.4175e+25, 3.7649e+25, 1e+26, 2.6561e+26, 7.0548e+26, 1.8738e+27, 4.977e+27, 1.3219e+28, 3.5112e+28, 9.326e+28, 2.4771e+29, 6.5793e+29, 1.7475e+30, 4.6416e+30, 1.2328e+31, 3.2745e+31, 8.6975e+31, 2.3101e+32, 6.1359e+32, 1.6298e+33, 4.3288e+33, 1.1498e+34, 3.0539e+34, 8.1113e+34, 2.1544e+35, 5.7224e+35, 1.5199e+36, 4.037e+36, 1.0723e+37, 2.848e+37, 7.5646e+37, 2.0092e+38, 5.3367e+38, 1.4175e+39, 3.7649e+39, 1e+40])
        self.p[0] = np.exp(interp1d(dsmax, np.log(cdmax))(dmax))
        # first guess td
        #-------------------
        # Finding from another type curve the dimensionless time corresponding
        # to a dimensionless drawdown of 0.5 for a given value of Cd
        ind = np.argsort(self.df.s)
        x = self.df.s[ind]
        y = self.df.t[ind]
        x, ind = np.unique(x, return_index=True)
        y = y[ind]
        t50 = interp1d(x,y)(0.5)
        td05 = np.array([0.00586770558742, 0.01538276678787, 0.03952949140660, 0.09694461204071, 0.21719603925390, 0.42501780012524, 0.71662305681555, 1.06363233029027, 1.43760222791051, 1.82028724290396, 2.20358418424062, 2.58408967048293, 2.96136937493074, 3.33482849790512, 3.70525943721090, 4.07263907432772, 4.43792575903868, 4.80135835601924, 5.16271293241688, 5.52325181572752, 5.88231715490175, 6.23931944071651, 6.59661383581317, 6.95210659097099, 7.30643224320023, 7.66115976998289, 8.01477682034437, 8.36756008586347, 8.71970765295706, 9.07135123700411, 9.42256498353144, 9.77337153570370, 10.12374618219556, 10.47361933182128, 10.82287789841075, 11.17136553273684, 11.52044344002349, 11.86974847357249, 12.21813252184930, 12.56530458198805, 12.91318067784498, 13.26149706560916, 13.60816009137916, 13.95493769518254, 14.30274614525803, 14.64826645321799, 14.99550293657282, 15.34193768499867, 15.68705068971524, 16.03430073477501, 16.37807076609380, 16.72575485388320, 17.06977548910524, 17.41657199079698, 17.76057059792890, 18.10691433716675, 18.45055957035689, 18.79684650709133, 19.13974751846824, 19.48634380902855, 19.82851525913239, 20.17529734012830, 20.51824259258036, 20.86351679873552, 21.20754050225741, 21.55073134371076, 21.89612895043074, 22.23763508467999, 22.58365010435056, 22.92687153886336, 23.26966547231196, 23.61487577912874, 23.95648343180061, 24.30112659032918, 24.64501565697978, 24.98550026910186, 25.33145439775837, 25.67448982989344, 26.01510452333986, 26.36095666948836, 26.70355761351395, 27.04411599835794, 27.38978772050444, 27.73233575282293, 28.07209819489549, 28.41796332106905, 28.76081111070927, 29.10077224808644, 29.44537021937825, 29.78884858313224, 30.12952033870633, 30.47177166764350, 30.81619529914736, 31.15787771659484, 31.49692855122661, 31.84248208061840, 32.18546561730872, 32.52586464152356, 32.86722245603525, 33.21179163617694])
        self.p[1] = t50 / np.exp(interp1d(np.log(cdmax),np.log(td05))(np.log(self.p[0])))
        # Second method
        #-------------------
        # When the method fails because the derivative is too high, Cd is fixed arbitrarily to 50.
        # Knowing that the dimensionless time td/cd = t/t0 = 1.7255 for sd=0.5 and Cd=50, we locate the time t corresponding to sd=0.5 and calculate t0=t/1.7255.
        if (np.isnan(self.p[0]) or np.isnan(self.p[1])):
            self.p[1] = t50/1.71255
            self.p[0] = 50

        return self.p


    def plot_typecurve(self, cD = 10 ** np.array([0, np.log10(5), np.log10(50), 5, 12, 40]), rD = 1):
        """
        Type curves of the Neuzil (1982) model
        """
        td = np.logspace(-3, 3)
        plt.figure(1)
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            self._coeff()
            sd = self._laplace_drawdown(td * cD[i], inversion_option='stehfest')
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.semilogx(td, sd, '-', color=color, label= 'C_D = {}'.format(cD[i]))
            ax.semilogx(dummy.der.t, np.abs(dummy.der.s), ':', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-3, 1e3))
        plt.ylim((0, 1e0))
        plt.grid('True')
        plt.legend()
        plt.show()

        plt.figure(2)
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            self._coeff()
            sd = self._laplace_drawdown(
                td * cD[i], inversion_option='stehfest')
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(td, sd, '-', color=color,
                        label='C_D = {}'.format(cD[i]))
            ax.loglog(dummy.der.t, np.abs(dummy.der.s), ':', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-3, 1e3))
        plt.ylim((1e-3, 1e0))
        plt.grid('True')
        plt.legend()
        plt.show()

    def rpt(self, fitmethod='lm', ttle='Neuzil (1982)', author='Author', filetype='pdf',
                reptext='Report_nsl', p=None):
        """
        Calculates the solution and reports graphically the results of the slug test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm'  # set Default

        if p is not None:
            self.p = p
            self.fitmethod = 'nofit'
        else:
            self.inversion_option = 'stehfest'

        self._coeff()
        self.fit()

        self.Transmissivity = self.T()
        self.Storativity = self.S()
        self.RadInfluence = self.RI_Guyonnet()

        self.model_label = 'Neuzil (1982)'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14, transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Effective Compressibility : {:3.2e} Pa^-1'.format(self.Ceff), fontsize = 14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Well radius : {:0.4g} m '.format(self.rw), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.7, 'Volume of test section: {:0.4g} m^3 '.format(self.Vs), fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.6, 'Hydraulic parameters :', fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Transmissivity T : {:3.2e} m²/s'.format(self.Transmissivity), fontsize=14,transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Well bore storage Sw : {:3.2e} '.format(self.Storativity), fontsize = 14,transform=plt.gcf().transFigure)
        fig.text(1.05, 0.45, 'Radius of investigation: {:3.2e} m'.format(self.RadInfluence[0]), fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.4, 'Fitting parameters :',fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'Wellbore storage cD : {:0.4g} '.format(self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} s'.format(self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'mean residual : {:0.2g} s'.format(self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, '2 standard deviation : {:0.2g} s'.format(self.sr), fontsize=14,transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, 'Root-mean-square : {:0.2g} s'.format(self.rms), fontsize=14,transform=plt.gcf().transFigure)


class Cooper(Neuzil):
    """
    Shut-in pulse or slug test with the Cooper (1967) solution

    The model is the same for Neuzil (1982) and Cooper (1967), but the
    calculation for transmissvity and storativity is different.

    :param rw: radius of the well
    :param rc: radius of the casing
    :param Ceff: effective Compressibility of test section
    :param Vs: Volume of the test section
    :param cD: dimensionless well bore storage coefficient
    :param p: solution vector
    :param df: pandas dataframe with two vectors named df.t and df.s for test time respective drawdown
    :param rho: density of water 1000 kg/m^3
    :param g: gravity 9.81 m/s^2
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
    :param fitmethod: see fit function for the various options
    :param inversion_option: 'stehfest' or 'dehoog'

    :Description:
    Conputes the normalized drawdown (Delta h / Delta h0) as a function of
    time with the Cooper et al. (1967) solution for a slug test in a
    homogeneous confined aquifer. The well is fully penetrating and the
    slug injection or withdrawal is instantaneous.

    The dimensionless well bore storage coefficient is:
    Cd = rc^2/(2 rw^2 S)

    The dimensionless time is related as followed:
    t0 = rc^2 / ( 2T )

    NB: Modified from Cooper et al. (1967) solution for a slug test.
    Note that in the original publication of Cooper et al.
    The dimensionless parameter was alpha, it is related to Cd by: alpha = 1 / (2 Cd)

    :Reference:
    Cooper, H.H.J., J.D. Bredehoeft, and I.S. Papadopulos.
    1967. Response of a finite-diameter well to an instantaneous charge of
    water. Water Resources Research 3, no. 1: 263-269.

    """

    def __init__(self, rw=None, rc=None, cD=None, df=None, p=None, Ceff=None, Vs=None, inversion_option=None):
        self.rw = rw
        self.rc = rc
        self.cD = cD
        self.p = p
        self.df = df
        self.Ceff = Ceff
        self.Vs = Vs
        self.der = None
        self.rho = 1000
        self.g = 9.81
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
        self.inversion_option=inversion_option
        self.fitcoeff = None


    def plot_typecurve(self, cD = 10 ** np.array([0, np.log10(5), np.log10(50), 5, 12, 40]), rD = 1):
        """
        Type curves of the Cooper (1967) model
        """
        td = np.logspace(-3, 3)
        plt.figure(1)
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            self._coeff()
            sd = self._laplace_drawdown(td * cD[i], inversion_option='stehfest')
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.semilogx(td, sd, '-', color=color, label= 'C_D = {}'.format(cD[i]))
            ax.semilogx(dummy.der.t, np.abs(dummy.der.s), ':', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-3, 1e3))
        plt.ylim((0, 1e0))
        plt.grid('True')
        plt.legend()
        plt.show()

        plt.figure(2)
        ax = plt.gca()
        for i in range(0, len(cD)):
            self.cd = cD[i]
            self._coeff()
            sd = self._laplace_drawdown(
                td * cD[i], inversion_option='stehfest')
            d = {'t': td, 's': sd}
            df = pda.DataFrame(data=d)
            dummy = ht.preprocessing(df=df)
            dummy.ldiff()
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(td, sd, '-', color=color,
                        label='C_D = {}'.format(cD[i]))
            ax.loglog(dummy.der.t, np.abs(dummy.der.s), ':', color=color)
        plt.xlabel('$t_D / C_D = 2Tt/r_C**2$')
        plt.ylabel('$s_D = 2*pi*T*s/Q$')
        plt.xlim((1e-3, 1e3))
        plt.ylim((1e-3, 1e0))
        plt.grid('True')
        plt.legend()
        plt.show()


    def rpt(self, fitmethod='lm', ttle='Cooper (1967)', author='Author', filetype='pdf',
                reptext='Report_cop', p=None):
        """
        Calculates the solution and reports graphically the results of the slug test

        :param option_fit: 'lm' or 'trf' or 'dogbox'
        :param ttle: Title of the figure
        :param author: Author name
        :param filetype: 'pdf', 'png' or 'svg'
        :param reptext: savefig name
        """
        if fitmethod is not None:
            self.fitmethod = fitmethod
        else:
            self.fitmethod = 'lm'  # set Default

        if p is not None:
            self.p = p
            self.fitmethod = 'nofit'
        else:
            self.inversion_option = 'stehfest'

        self._coeff()
        self.fit()

        self.Transmissivity = 0.5 * self.rc ** 2/ self.p[1]
        self.Storativity = 0.5 / self.p[0]*(self.rc/self.rw) ** 2


        self.model_label = 'Cooper (1967)'
        test = ht.preprocessing(df=self.df)
        self.der = test.ldiffs()

        self.ttle = ttle
        fig = log_plot(self)

        fig.text(0.125, 1, author, fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(0.125, 0.95, ttle, fontsize=14,
                 transform=plt.gcf().transFigure)

        fig.text(1, 0.85, 'Test Data : ', fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.8, 'Well radius : {:0.4g} m '.format(
            self.rw), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.75, 'Casing radius : {:0.4g} m '.format(
            self.rc), fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.6, 'Hydraulic parameters :',
                 fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.55, 'Transmissivity T : {:3.2e} m²/s'.format(
            self.Transmissivity), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.5, 'Well bore storage Sw : {:3.2e} '.format(
            self.Storativity), fontsize=14, transform=plt.gcf().transFigure)

        fig.text(1, 0.4, 'Fitting parameters :', fontsize=14,
                 transform=plt.gcf().transFigure)
        fig.text(1.05, 0.35, 'Wellbore storage cD : {:0.4g} '.format(
            self.p[0]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.3, 'intercept t0 : {:0.2g} s'.format(
            self.p[1]), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.25, 'mean residual : {:0.2g} s'.format(
            self.mr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.2, '2 standard deviation : {:0.2g} s'.format(
            self.sr), fontsize=14, transform=plt.gcf().transFigure)
        fig.text(1.05, 0.15, 'Root-mean-square : {:0.2g} s'.format(
            self.rms), fontsize=14, transform=plt.gcf().transFigure)


class CDMHeat(AnalyticalInterferenceModels):
    """
    Heat tracer test implementation 
    The solution accounts for ....

    :param rw: radius of the injection well
    :param r: radius between injection and observation well 
    :param rID: rw/r 
    :param Q: flow rate m^3/s
    :param H: aperture of fracture m
    :param alpha: dispersivity m
    :param Dm: thermal diffusivity of matrix m^2/s
    :param cm: specfic heat matrix J/K/kg
    :param rhom: density of rock mass kg/m^3
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
    :param detailled_p: detailled solution struct from the fit function
    :param fitmethod: see fit function for the various options
    :param inversion_option: 'stehfest' or 'dehoog'   
    """

    def __init__(self, rw=1, r=10, Q=None, H=None, cm=None, rhom=None, alpha=None, Dm=None, df=None, p=None, inversion_option=None):
        self.rw = rw
        self.r = r
        self.rID = rw / r
        self.Q = Q
        self.H = H
        self.alpha = alpha
        self.cm = cm
        self.rhom = rhom
        self.Dm = Dm
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
        self.detailled_p = None
        self.fitmethod = None
        self.fitbnds = None
        self.inversion_option=inversion_option
        self.fitcoeff = None
        self.inversion_M = None

    def __call__(self, t):
        s = self._laplace_drawdown(t)
        return s

    def dimensionless_laplace(self, pd):
        p = (1+self.sigma) / np.sqrt(pd*self.tr) * self.tb * pd
        sigmaf = (2.*p)/(self.Pe**2*(1-self.rID**2))
        yI = self.rID * self.Pe + 1/(4*sigmaf)
        y = self.Pe + 1/(4*sigmaf)
        ai, aip, bi, bip = airy(sigmaf**(1/3)*y)
        aiI, aiP, biI, biP = airy(sigmaf**(1/3)*yI)
        return ((np.exp(y-yI)/2)*ai) / (0.5 * aiI - sigmaf ** (1/3) * aiP)

    def plot_typecurve(self):
        td = np.logspace(-1, 7, 1000)
        self.tb = 200 #advective time
        self.tr = 0.0063
        Pe = [1, 10, 100] #Peclet number
        self.rID = 0.01
        self.sigma = 1
        fig, ax = plt.subplots(1,1)
        self._coeff()
        for i in range(0, len(Pe)):
            self.Pe = Pe[i]
            sd = abs(self._laplace_drawdown(td, inversion_option='stehfest'))
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(td, sd, '-', color=color, label= 'Pe = {}'.format(Pe[i]))
            print(sd)
        plt.xlabel('$t_D$')
        plt.ylabel('$C_D$')
        plt.title('CMD Typecurves')
        plt.xlim((0, 1e7))
        plt.ylim((0, 4e-6))
        plt.grid('True')
        plt.legend()
        plt.show()
