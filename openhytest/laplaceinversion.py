#    Copyright (C) 2020 by
#    Nathan Dutler <nathan.dutler@unine.ch>
#    All rights reserved.
#    MIT license.


"""
Laplace inversion toolbox
**************************


License
---------
Released under the MIT license:
   Copyright (C) 2019 openhytest Developers
   Nathan Dutler <nathan.dutlern@unine.ch>
   Philippe Renard <philippe.renard@unine.ch>
   Bernard Brixel <bernard.brixel@erdw.ethz.ch>

"""

class invertlaplace():

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

class InverseLaplaceTransform(object):
    r"""
    Inverse Laplace transform methods are implemented using this
    class and with multiprocessing pool for inversion.
    """

    def __init__(self,ctx):
        self.ctx = ctx

class Stehfest(InverseLaplaceTransform):

    def calc_laplace_parameter(self,t,**kwargs):
        r"""
        The Gaver-Stehfest method is a discrete approximation of the
        Widder-Post inversion algorithm, rather than a direct
        approximation of the Bromwich contour integral.
        The method abscissa along the real axis, and therefore has
        issues inverting oscillatory functions (which have poles in
        pairs away from the real axis).
        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.
        .. math ::
            p_k = \frac{k \log 2}{t} \qquad 1 \le k \le M
        """

        # required
        # ------------------------------
        # time of desired approximation
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38*self.degree)
        else:
            self.dps_goal = int(2.93*self.ctx.dps)
            self.degree = max(16,self.dps_goal)

        # _coeff routine requires even degree
        if self.degree%2 > 0:
            self.degree += 1

        M = self.degree

        # this is adjusting the dps of the calling context
        # hopefully the caller doesn't monkey around with it
        # between calling this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        self.V = self._coeff()
        self.p = self.ctx.matrix(self.ctx.arange(1,M+1))*self.ctx.ln2/self.t

        # NB: p is real (mpf)

    def _coeff(self):
        r"""Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision"""

        M = self.degree
        M2 = int(M/2) # checked earlier that M is even

        V = self.ctx.matrix(M,1)

        # Salzer summation weights
        # get very large in magnitude and oscillate in sign,
        # if the precision is not high enough, there will be
        # catastrophic cancellation
        for k in range(1,M+1):
            z = self.ctx.matrix(min(k,M2)+1,1)
            for j in range(int((k+1)/2),min(k,M2)+1):
                z[j] = (self.ctx.power(j,M2)*self.ctx.fac(2*j)/
                        (self.ctx.fac(M2-j)*self.ctx.fac(j)*
                         self.ctx.fac(j-1)*self.ctx.fac(k-j)*
                         self.ctx.fac(2*j-k)))
            V[k-1] = self.ctx.power(-1,k+M2)*self.ctx.fsum(z)

        return V

    def calc_time_domain_solution(self,fp,t,manual_prec=False):
        r"""Compute time-domain Stehfest algorithm solution.
        .. math ::
            f(t,M) = \frac{\log 2}{t} \sum_{k=1}^{M} V_k \bar{f}\left(
            p_k \right)
        where
        .. math ::
            V_k = (-1)^{k + N/2} \sum^{\min(k,N/2)}_{i=\lfloor(k+1)/2 \rfloor}
            \frac{i^{\frac{N}{2}}(2i)!}{\left(\frac{N}{2}-i \right)! \, i! \,
            \left(i-1 \right)! \, \left(k-i\right)! \, \left(2i-k \right)!}
        As the degree increases, the abscissa (`p_k`) only increase
        linearly towards `\infty`, but the Stehfest coefficients
        (`V_k`) alternate in sign and increase rapidly in sign,
        requiring high precision to prevent overflow or loss of
        significance when evaluating the sum.
        **References**
        1. Widder, D. (1941). *The Laplace Transform*. Princeton.
        2. Stehfest, H. (1970). Algorithm 368: numerical inversion of
           Laplace transforms. *Communications of the ACM* 13(1):47-49,
           http://dx.doi.org/10.1145/361953.361969
        """

        # required
        self.t = self.ctx.convert(t)

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter(), so is already
        # a list or matrix of mpmath 'mpf' types

        result = self.ctx.fdot(self.V,fp)*self.ctx.ln2/self.t

        # setting dps back to value when calc_laplace_parameter was called
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        # ignore any small imaginary part
        return result.real
