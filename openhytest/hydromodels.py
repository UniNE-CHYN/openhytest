# -*- coding: utf-8 -*-

# _____________________
# 
# Copyright (C) 2021
# Nathan Dutler <nathan.dutler@unine.ch>
# Philippe Renard <philippe.renard@unine.ch>
# Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#
# All rights reserved
# MIT license
# _____________________
# _____________________
#
#      Flow models 
# _____________________
# _____________________
#
# This file contains numerical implementations of analytical flow models
# used to interpret well tests, both in porous and fractured media.
# 
# Analytical solutions are programed following an oriented-object approach, 
# since many models share similar properties. For the case of solutions 
# derived in the Laplace space, two well known inversion algorithms are
# provided, the Stefhest (1970) and de Hoog (1982) methods, to invert parameters 
# back in real space
#
# Basic plotting utilities are also provided to guide well test analyses, 
# from initial interpretation to final reporting stages.

import sys
import pandas as pd
import numpy as np 
import mpmath as mp 
import matplotlib.pyplot as plt 

from scipy.special import expn as E1 
from scipy.special import gamma, gammaincc, factorial, kv 
from scipy.optimize import least_squares, Bounds 
from scipy.interpolate import interp1d 

from hytestutils import htu

# Parent class 
# _____________________
class AnalyticalInterferenceModels():
    """Parent class providing the basic methods and properties inherited by interference (cross-hole) flow models. 

    Attributes
    ----------
    Q : float
        pumping or injection flow rate
    r : float
        borehole radius
    Rd : float
        radius of influence
    p : ???
        ???
    df : ???
        ????

    Methods
    -------
    T()
        Calculates transmissivity
    S()
        Calculates storativity

    trial(p=None, inversion_option=None)
        Displays field data and calculated solution together.

    fit(fitmethod=None, fitcoeff=None)
        Fit parameter(s) of a given model.

    dehoog(Fp, td, alpha=0, tol=1e-9, M=20)
        Numerical Laplace inversion using the dehoog method.
    
    stehfest(Fp, td)
        Numerical Laplace inversion using the Stefhest method.
    """
    
    def __init__(self, Q=None, r=None, Rd=None, p=None, df=None):
        """
        Parameters
        ----------
        Q : float
            pumping or injection flow rate
        r : float
            borehole radius
        Rd : float
            radius of influence
        p : ???
            ???
        df : ???
            ????
        """

        self.Q = Q
        self.r = r
        self.Rd = Rd
        self.p = p
        self.df = df
        self.inversion_M = None
        self.inversion_V = None
        self.inversion_s = None

    def _dimensionless_time(self, t):
        """Calculates dimensionless time.

        Parameters
        ----------
        t : float
            elapsed time

        Returns
        -------
        array
            an array containing dimensionless time
        """
        
        return (t / (0.5628 * self.p[1])) * 0.25

    def _dimensional_drawdown(self, sd):  # SHOULD IT BE DIMENSIONLESS INSTEAD OF DIMENSIONAL?
        """Calculates the dimensional drawdown.
        
        Parameters
        ----------
        sd : float
            drawdown or pressure build up

        Returns
        -------
        array
            an array containing the dimensionless drawdown or pressure buildup
        """

        return (sd * self.p[0] * 2) / 2.302585092994046

    def _dimensional_flowrate(self, qd):  # SHOULD IT BE DIMENSIONLESS INSTEAD OF DIMENSIONAL?
        """Calculates the dimensional flow rate."""

        return qd * np.log(10) / 2.0 / self.p[0]

    def _laplace_drawdown(self, td, inversion_option=None):
        """Alternative calculation with Laplace inversion.

        Parameters
        ----------
        td : float 
            dimensionless time array
        inversion_option : str, optional
            option to pick either the Stehfest or dehoog method

        Returns
        -------
        sd : float
            dimensionless drawdown
        """

        if inversion_option is not None:
            self.inversion_option = inversion_option
        
        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace, td)
        
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace, td)  
        
        return sd

    def _laplace_drawdown_types(self, td, inversion_option=None): 
        """Alternative calculation with Laplace inversion.

        Parameters
        ----------
        td : float 
            dimensionless time array
        inversion_option : str, optional
            option to pick either the Stehfest or dehoog method

        Returns
        -------
        sd : float
            dimensionless drawdown
        """
        
        if inversion_option is not None:
            self.inversion_option = inversion_option
        
        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace_types, td)
        
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace_types, td)  
        
        return sd
  
    def _laplace_drawdown_derivative(self, td, inversion_option=None): 
        """Alternative calculation with Laplace inversion.

        Parameters
        ----------
        td : float
            dimensionless time
        inversion_option : str, optional
            opion to pick the inversion method (Stehfest or dehoog)
        
        Returns
        -------
        dd : float
            dimensionless drawdown derivative
        """
        
        if inversion_option is not None:
            self.inversion_option = inversion_option
        
        if self.inversion_option == 'stehfest':
            sd = self.stehfest(self.dimensionless_laplace_derivative, td)
        
        elif self.inversion_option == 'dehoog':
            sd = self.dehoog(self.dimensionless_laplace_derivative, td)  
        
        return sd

    def _coeff(self, fitcoeff=12):
        """Calculates the coefficent for the stehfest method.

        Parameters
        ----------
        fitcoeff : int
            number of coefficients for inversion (default=12)

        Returns
        -------
        self.inversion_V : int
            inversion coefficent for stehfest
        self.inversion_M : int
            number of coefficients for inversion
        V : int
            inversion coefficent
        """

        if fitcoeff != 12:
            M = np.int(self.fitcoeff)
        
        else:
            M = fitcoeff # Default

        if M % 2 > 0: # Check if M is even
            M = M+1
        
        V = np.zeros(M)

        for i in range(1, M+1):
            vi = 0
            
            for k in range(np.int((i+1)/2), np.int(np.min([i,M/2]))+1):
                vi = vi + (k**(M/2) * factorial(2*k)) / (factorial(np.int(M/2-k)) * factorial(k) * factorial(k-1) * factorial(i-k) * factorial(2*k-i))
            
            V[i-1] = (-1)**((M/2)+i)*vi
        
        self.inversion_V = V
        self.inversion_M = M
        
        return V

    def stehfest(self, Fp, td):  # Need to add the definition of the keyword arguments
        """Numerical Laplace inversion with the Stefhest method.

        Parameters
        ----------
        Fp : ???
            ???
        td : float
            dimensionless time

        Returns
        -------
        self.inversion_s : float
            calculated drawdown in time domain
        S : float
            the calculated drawdown in time domain

        Ref.
        ---- 
        Widder, D. (1941). The Laplace Transform. Princeton.
        Stehfest, H. (1970). Algorithm 368: numerical inversion of Laplace transforms. 
        Communications of the ACM 13(1):47-49, http://dx.doi.org/10.1145/361953.361969
        """

        p = np.zeros([self.inversion_M, np.size(td)])
        
        for i in range(1, self.inversion_M+1):
            p[i-1] = i*np.log(2)/td
        
        uu = Fp(p)
        VV = np.repeat(self.inversion_V, np.size(td)).reshape(self.inversion_M, np.size(td))
        su = np.multiply(VV, uu)
        s = np.log(2)/td*sum(su)
        self.inversion_s = s
        
        return s

    def dehoog(self, Fp, td, alpha=0, tol=1e-9, M=20):  # Is s instead f? Carefull with the docstring doc
        """Numerical Laplace inversion with the dehoog method.

        de Hoog et al's quotient difference method with accelerated convergence for the continued fraction expansion

        Modification:
        The time vector (td) is split in segments of equal magnitude which are
        inverted individually. This gives a better overall accuracy.
        
        Returns
        -------
        self.inversion_s : float
            the calculated drawdown in time domain
        f : float
            the calculated drawdown in time domain

        Ref.
        ---- 
        de Hoog, F. R., Knight, J. H., and Stokes, A. N. (1982). An improved
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
                gamma = alpha-np.log(tol)/(2*T)
                run = np.linspace(0,2*M, 2*M+1)
                p = gamma + 1j * np.pi * run / T    
                a = Fp(p)      # evaluate function
                a[0] = a[0] / 2 # zero term is halved
                
                # Build up e and q tables.
                e = np.zeros((2*M+1, M+1), dtype=complex)
                q = np.zeros((2*M, M+1), dtype=complex)
                q[:,1] = a[1:2*M+1]/a[0:2*M]

                for r in np.arange(1,M+1):
                    e[0:2*(M-r)+1,r] = q[1:2*(M-r)+2,r] - q[0:2*(M-r)+1,r] + e[1:2*(M-r)+2,r-1]
                    if r < M:
                        rq = r + 1
                        q[0:2*(M-rq)+2,rq] = q[1:2*(M-rq)+3,rq-1]*e[1:2*(M-rq)+3,rq-1]/e[0:2*(M-rq)+2,rq-1]
                
                # Build up d vector  
                d = np.zeros((2*M+1,1), dtype=complex)
                d[0] = a[0]
                d[1:2*M:2] = np.vstack(-q[0,1:M+1])
                d[2:2*M+1:2] = np.vstack(-e[0,1:M+1])
                
                # Build up A and B matrix
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
        """Calculates Transmissivity."""
        
        return np.log(10) / np.pi/4 * self.Q / self.p[0]

    def S(self):
        """ Calculates Storativity."""

        return 2.2458394 * self.T() * self.p[1] / self.r ** 2

    def trial(self, p=None, inversion_option=None):  # loglog included: derivatives are missing at the moment.
        """Displays field data and calculated solution together.

        The function produces a graph that superposes data and model results. This can be used to test graphically the quality of a
        fit, or to adjust manually the parameters of a model until a satisfactory fit is obtained.

        Parameters
        ----------
        p : array
            a solution vector can be initialized
        inversion_option : str, optional
            ?? (Default de Hoog (1982))
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
        """Fit the model parameter of a given model.

        The function optimizes model parameters to fit observations. The fit is obtained by an iterative
        non linear least square procedure (an initial guess is therefore required, which will then be iterativly modified
        until a local minimum is obtained.

        Parameters
        ----------
        fitmethod : str, optional
            Levenberg-Marquard (lm is default), Trust Region Reflection algorithm (trf) or dogbox using least-squares implementation from scipy-optimize or use nofit to caculate only the statistic 
        fitcoeff : int
            Number of coefficent defined for Laplace inversions
        
        Returns
        -------
        res_p.x : array
             solution vector p
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

        # Cost function
        def fun(p, t, s):
            self.p = p
            return s.to_numpy() - self.__call__(t.to_numpy())

        if self.fitmethod == 'lm':
            # Levenberg-Marquardt algorithm (Default). 
            # Does not handle bounds and sparse Jacobians. 
            # Provides usually the most efficient method for small, unconstrained problems.
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
            raise Exception('Choose a fitting method: lm, trf and dogbox')
        
        # Define regular points to plot the calculated drawdown
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