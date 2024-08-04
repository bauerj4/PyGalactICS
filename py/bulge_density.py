import numpy as np
from scipy.special import gamma, gammainc, gammaincc


class BulgeDensObj:
    def __init__(
        self,
        nnn: float = None,
        re: float = None,
        v0bulge=None,
        abulge: float = None,
        rho0: float = None,
        pp=None,
        butt=None,
        **kwargs,
    ):
        """
        Initialize the Bulge Density Object. Note that the bulge is not
        truncated like other components.

        Parameters
        ----------
        nnn : float
            The Sersic index.
        re : float
            Effective radius of the bulge.
        v0bulge : float
            Normalization constant for the bulge.
        abulge : float
            Characteristic parameter for the bulge.
        rho0 : float, optional
            Normalization density. If None, it will be calculated.
        pp : float, optional
            Sersic parameter. If None, it will be calculated.
        butt : callable, optional
            Function to compute the Sersic parameter. If None, it will be calculated.
        """
        self.nnn = nnn
        self.re = re
        self.v0bulge = v0bulge
        self.abulge = abulge
        self.rho0 = rho0
        self.pp = pp
        self.butt = butt

        if self.pp is None:
            self.pp = 1.0 - 0.6097 / self.nnn + 0.05463 / (self.nnn * self.nnn)
        if self.butt is None:
            self._calculate_sersic_parameters()

    def _calculate_sersic_parameters(self):
        """
        Calculate the Sersic parameters.
        """
        comnen = self.nnn
        self.re = self.abulge

        if self.nnn > 0.50:
            butt1 = 0.6 * (2.0 * self.nnn)
        else:
            butt1 = 1.0e-4

        butt2 = 1.20 * (2.0 * self.nnn)
        buttacc = 1.0e-4

        def funcbutt(b):
            abe = 2.0 * comnen
            return gammaincc(abe, b) - 0.5

        self.butt = self._rtbiss(funcbutt, butt1, butt2, buttacc, comnen)

        # Calculate rho0 if not provided
        if self.rho0 is None:
            self.rho0 = (self.v0bulge**2) / (
                4
                * np.pi
                * self.re**2
                * self.nnn
                * self.butt ** (self.nnn * (self.pp - 2))
                * np.exp(gamma(self.nnn * (2 - self.pp)))
            )

    def _rtbiss(self, func, x1, x2, xacc, comnen):
        """
        Bisection method to find the root of a function.

        Parameters
        ----------
        func : callable
            Function for which to find the root.
        x1 : float
            Lower bound of the interval.
        x2 : float
            Upper bound of the interval.
        xacc : float
            Desired accuracy of the result.
        comnen : float
            Parameter for the function.

        Returns
        -------
        float
            Estimated root.
        """
        JMAX = 100
        fmid = func(x2)
        f = func(x1)

        if f * fmid >= 0:
            raise ValueError("Root must be bracketed in rtbiss")

        if f < 0:
            x = x1
            dx = x2 - x1
        else:
            x = x2
            dx = x1 - x2

        for _ in range(JMAX):
            dx *= 0.5
            xmid = x + dx
            fmid = func(xmid)
            if fmid <= 0:
                x = xmid
            if abs(dx) < xacc or fmid == 0:
                return xmid

        raise ValueError("Too many bisections in rtbiss")

    def sersicdens(self, rad):
        """
        Calculate the Sersic density profile.

        Parameters
        ----------
        rad : float
            Radius at which to calculate the density.

        Returns
        -------
        float
            Density at the specified radius.
        """
        u = rad / self.re
        un = u ** (1.0 / self.nnn)
        return self.rho0 * (u ** (-self.pp)) * np.exp(-self.butt * un)

    def sersicdensprime(self, rad):
        """
        Calculate the derivative of the Sersic density profile.

        Parameters
        ----------
        rad : float
            Radius at which to calculate the derivative.

        Returns
        -------
        float
            Derivative of the density at the specified radius.
        """
        u = rad / self.re
        un = u ** (1.0 / self.nnn)
        return (
            -self.sersicdens(rad)
            / self.re
            * (self.pp * self.nnn + self.butt * un)
            / (self.nnn * u)
        )

    def sersicdens2prime(self, rad):
        """
        Calculate the second derivative of the Sersic density profile.

        Parameters
        ----------
        rad : float
            Radius at which to calculate the second derivative.

        Returns
        -------
        float
            Second derivative of the density at the specified radius.
        """
        u = rad / self.re
        un = u ** (1.0 / self.nnn)
        return (
            self.sersicdens(rad)
            / (self.re**2)
            * (
                (self.pp * self.nnn) ** 2
                + self.pp * self.nnn * self.nnn
                + 2 * self.pp * self.butt * self.nnn * un
                + self.butt * un * (self.nnn - 1)
                + (self.butt * un) ** 2
            )
            / (self.nnn * u) ** 2
        )
