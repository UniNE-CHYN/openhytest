#    Copyright (C) 2019 by
#    Philippe Renard <philippe.renard@unine.ch>
#    Nathan Dutler <nathan.dutler@unine.ch>
#    Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#    All rights reserved.
#    MIT license.


"""
Analytical model classes
========================
The different analytical model classes are implemented to fit the observations given as dataframe in time (t) and drawdown (s).
License
-------
Released under the MIT license:
   Copyright (C) 2019 openhytest Developers
   Philippe Renard <philippe.renard@unine.ch>
   Nathan Dutler <nathan.dutlern@unine.ch>
   Bernard Brixel <bernard.brixel@erdw.ethz.ch>
"""

import numpy as np
from scipy.special import expn as E1
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import mpmath as mp


# Utilities

def get_logline(df):   
    logt = np.log10( df.t ).values
    Gt = np.array( [logt, np.ones(logt.shape)] )
    p = np.linalg.inv( Gt.dot(Gt.T) ).dot(Gt).dot(df.s) 
    p[1] = 10**( -p[1]/p[0] )
    return p
  
# Parent generic class

class AnalyticalModels():
    def __init__(self,Q=None,r=None,Rd=None):
        self.Q = Q
        self.r = r
        self.Rd = Rd
    
    def _dimensionless_time(self,p,t):
        return (t/(0.5628*p[1])) * 0.25
    
    def _dimensional_drawdown(self,p,sd):
        return (sd*p[0]*2)/2.302585092994046
 
    def _laplace_drawdown(self, td, option='Stehfest'): #default stehfest
        return map( lambda x: mp.invertlaplace(self.dimensionless_laplace, x, method = option, dps = 10, degree = 12), td)
    
    def _laplace_drawdown_derivative(self,td, option='Stehfest'): #default stehfest
        return map( lambda x: mp.invertlaplace(self.dimensionless_laplace_derivative, x, method = option, dps = 10, degree = 12), td) 

    def __call__(self,t):
        print("Warning - undefined")
        return None
    
    def T(self,p):
        return 0.1832339 * self.Q / p[0]
    
    def S(self,p):
        return 2.2458394*self.T(p)*p[1]/self.r**2  
    
    def trial(self, p, df): #loglog included: derivatives are missing at the moment.
        figt = plt.figure()
        ax1 = figt.add_subplot(211)
        ax2 = figt.add_subplot(212)
        ax1.loglog(df.t, self.__call__(p,df.t), df.t, df.s,'o' )
        ax1.set_ylabel('s')
        ax1.grid()
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        ax2.semilogx(df.t, self.__call__(p,df.t), df.t, df.s,'o' )
        ax2.set_ylabel('s')
        ax2.set_xlabel('t')
        ax2.grid()
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.show()
        print( 'T = ', self.T(p) ,'m2/s')
        print( 'S = ', self.S(p) ,'-')
        print( 'Ri = ',self.RadiusOfInfluence(p,df.t),'m' )
        
        
    def fit(self, p,  df, option='lm', output='all'):   
        t = df.t
        s = df.s
        # costfunction
        def fun(p, t, s):
            return np.array(s) - self.__call__(p,t)

        if option == 'lm':
            # Levenberg-Marquard -- Default
            res_p = least_squares(fun, p, args=(t,s), method='lm', xtol=1e-10, verbose=1)      
        elif option == 'trf':
            # Trust Region Reflective algorithm
            res_p = least_squares(fun, p, jac='3-point', args=(t,s), method='trf', verbose=1)
        else: 
            raise Exception('Specify your option')

        if output == 'all': #-- Default
            #define regular points to plot the calculated drawdown
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t)-1]),  num = len(t), endpoint = True, base = 10.0, dtype = np.float64)
            sc = self.__call__(res_p.x,tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun**2))
            return res_p.x, tc, sc, mr, sr, rms
        elif output == 'p':
            return res_p.x
        elif output  == 'Detailled':
            tc = np.logspace(np.log10(t[0]), np.log10(t[len(t)-1]),  num = len(t), endpoint = True, base = 10.0, dtype = np.float64)
            sc = self.__call__(res_p.x,tc)
            mr = np.mean(res_p.fun)
            sr = 2 * np.nanstd(res_p.fun)
            rms = np.sqrt(np.mean(res_p.fun**2))
            return res_p, tc, sc, mr, sr, rms
        else:
            raise Exception('The output needs to specified: p or all')    
            
# Derived daughter classes

class theis(AnalyticalModels):
    
    def dimensionless(self, td):
        return 0.5*E1(1,0.25/td)
    
    def dimensionless_logderivative(self, td):
        return 0.5*np.exp(-0.25/td)
    
    def dimensionless_laplace(self, pd):
        return 1/pd*mp.besselk(0, mp.sqrt(pd))
    
    def dimensionless_laplace_derivative(self, pd):
        return 0.5*mp.besselk(1, mp.sqrt(pd))/mp.sqrt(pd)
        
    def __call__(self, p, t):
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless( td )
        s = self._dimensional_drawdown(p, sd)
        return s
    
    def guess_params(self, df):
        n = len(df)/3
        return get_logline(df[df.index>n])
    
    def RadiusOfInfluence(self,p,t):
        return 2*np.sqrt( self.T(p) * t[len(t)-1] / self.S(p) ) 
    
    def plot_typecurve(self):
        td = np.logspace(-1, 4)
        sd = self.dimensionless(td)
        dd = self.dimensionless_logderivative(td)

        plt.loglog(td,sd,td,dd,'--')
        plt.xlabel('$t_D$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-1,1e4))
        plt.ylim((1e-2,10))
        plt.grid('True')
        plt.legend(['Theis','Derivative'])
        plt.show()
        
class theis_noflow(AnalyticalModels):
    
    def dimensionless(self, td, Rd):
        ths = theis()
        return ths.dimensionless(td) + ths.dimensionless(td/Rd**2)
    
    def dimensionless_logderivative(self, td, Rd):
        ths = theis()
        return ths.dimensionless_logderivative(td) + ths.dimensionless_logderivative(td/Rd**2)
    
    def dimensionless_laplace(self, pd):
        return 1/pd*mp.besselk(0, mp.sqrt(pd)) + 1/(pd)*mp.besselk(0, mp.sqrt(pd)*self.Rd)
    
    def dimensionless_laplace_derivative(self, pd):
        return 0.5*mp.besselk(1, mp.sqrt(pd))/mp.sqrt(pd) + 0.5*mp.besselk(1, mp.sqrt(pd)*self.Rd)/mp.sqrt(pd)*self.Rd 
     
    def __call__(self, p, t):
        Rd =  np.sqrt(p[2] / p[1])
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless(td, Rd) 
        s = self._dimensional_drawdown(p, sd)
        return s
    
    def guess_params(self, df):
        n = len(df)/4
        p_late = get_logline(df[df.index>n])      
        p_early = get_logline(df[df.index<2*n])
        return np.array([p_late[0]/2, p_early[1], p_late[1]**2/p_early[1]])

    def RadiusOfInfluence(self,p,t):
        return np.sqrt(2.2458394* self.T(p) * p[2] / self.S(p) ) 
    
    def plot_typecurve(self, Rd=np.array([1.3, 3.3, 10, 33])):
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            sd = self.dimensionless(td, Rd[i])
            dd = self.dimensionless_logderivative(td, Rd[i]) 
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label = Rd[i])
            plt.loglog(td, dd, '-.', color=color)                     
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-2, 20))
        plt.grid('True')
        plt.legend()
        plt.show()
        
#class theis_superposition(AnalyticalModels):
        
class theis_constanthead(AnalyticalModels):

    def dimensionless(self, td, Rd):
        ths = theis()
        return ths.dimensionless(td) - ths.dimensionless(td/Rd**2)
    
    def dimensionless_logderivative(self, td, Rd):
        ths = theis()
        return ths.dimensionless_logderivative(td) - ths.dimensionless_logderivative(td/Rd**2)  
    
    def dimensionless_laplace(self, pd):
        return 1/pd*mp.besselk(0, mp.sqrt(pd)) - 1/(pd)*mp.besselk(0, mp.sqrt(pd)*self.Rd)

    def dimensionless_laplace_derivative(self, pd):
        return 0.5*mp.besselk(1, mp.sqrt(pd))/mp.sqrt(pd) - 0.5*mp.besselk(1, mp.sqrt(pd)*self.Rd)/mp.sqrt(pd)*self.Rd 
    
    def __call__(self, p, t):
        Rd =  np.sqrt(p[2] / p[1])
        td = self._dimensionless_time(p, t)
        sd = self.dimensionless(td, Rd) 
        s = self._dimensional_drawdown(p, sd)
        return s
    
    def guess_params(self, df):
        n = len(df)/4
        p_late = get_logline(df[df.index>n]) 
        p_early= get_logline(df[df.index<2*n]) 
        return np.array([p_early[0], p_early[1], 2*p_late[1]*p_early[1]**2/p_late[0]**2]) #CHECK THIS
    
    def RadiusOfInfluence(self,p,t):
        return np.sqrt(2.2458394*self.T(p)*p[2]/self.S(p))
    
    def plot_typecurve(self, Rd=np.array([1.5, 3, 10, 30])):
        td = np.logspace(-2, 5)
        ax = plt.gca()
        for i in range(0, len(Rd)):
            sd = self.dimensionless(td, Rd[i])
            dd = self.dimensionless_logderivative(td, Rd[i]) 
            color = next(ax._get_lines.prop_cycler)['color']
            plt.loglog(td, sd, '-', color=color, label = Rd[i])
            plt.loglog(td, dd, '-.', color=color)                     
        plt.xlabel('$t_D / r_D^2$')
        plt.ylabel('$s_D$')
        plt.xlim((1e-2, 1e5))
        plt.ylim((1e-2, 20))
        plt.grid('True')
        plt.legend()
        plt.show()

class special(AnalyticalModels):
    
    def goodman_discharge(self, l, T, r0):
        return 2*np.pi*T*l/(np.log(2*l/r0))
    
    def calc_sl_du(self, Rd):
        sldu = []
        for i in range(0, np.size(Rd)):        
            if Rd[i] <= 1:
                Rd[i] = 1.00001  
            sldu.append(np.log(Rd[i]**2)/((Rd[i]**2-1)*Rd[i]**(-2*Rd[i]**2/(Rd[i]**2-1))))
        return sldu
    
    def calc_inverse_sl_du(self, fri):
        if fri < 2.71828182850322:
            print('Problem in the inversion of Rd: calc_sl_du')
            Rd = 1.000001   
        else:
            Rd = np.exp(fri/2)
            if Rd < 50:
                y = np.linspace(1.00001, 60, 2000)
                x = self.calc_sl_du(y)
                frd = interpolate.interp1d(x,y)
                Rd = frd(fri)
        return Rd  