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
