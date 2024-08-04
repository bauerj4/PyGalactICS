import numpy as np

from truncation import eerfc, eerfcprime, eerfc2prime
from harmonic_integration import simpson_harm_int


class HaloDensObj:
    """
    A class to compute the halo density and its derivatives using the NFW profile with truncation.
    """

    def __init__(
        self,
        r_s: float = 10.0,
        v0: float = 100.0,
        cusp: float = 1.0,
        chalo: float = 100.0,
        dr_trunc: float = 10.0,
        truncation_region_width=4.0,
        eps: float = 0.0001,
        **kwargs,
    ) -> None:
        """
        Initializes the HaloDensObject with the given parameters.

        Parameters
        ----------
        r_s : float
            The scale radius of the halo.
        v0 : float
            The characteristic velocity of the halo in units of 100 km / s
        cusp : float, optional
            The cusp parameter for the NFW profile (default is 1.0).
        chalo : float, optional
            The truncation parameter of the halo (default is 1.0).
        dr_trunc : float, optional
            The truncation scale or width (default is None, which implies using `r_s` or another default value).
        truncation_region_width : float
            The truncation region width
        eps : float
            Small radius for density calculation (kpc)
        """
        self.r_s = r_s
        self.v0 = v0
        self.cusp = cusp
        self.chalo = chalo
        self.dr_trunc = (
            dr_trunc if dr_trunc is not None else r_s
        )  # Default to scale radius if not specified
        self.halo_const = 2 ** (1.0 - cusp) * v0**2 / (4 * np.pi * r_s**2)
        self.truncation_region_width = truncation_region_width
        self.eps = eps

    def density(self, rad: np.ndarray) -> np.ndarray:
        """
        Computes the halo density at given radial distances.

        Parameters
        ----------
        rad : np.ndarray
            The radial distances at which to compute the halo density.

        Returns
        -------
        np.ndarray
            The halo density values at the given radial distances.
        """
        s = rad / self.r_s
        density = self.halo_const / (s**self.cusp * (1 + s) ** (3 - self.cusp))
        return density * eerfc(
            rad, self.chalo, self.dr_trunc, self.truncation_region_width
        )

    def density_prime(self, rad: np.ndarray) -> np.ndarray:
        """
        Computes the first derivative of the halo density with respect to radial distance.

        Parameters
        ----------
        rad : np.ndarray
            The radial distances at which to compute the first derivative of the halo density.

        Returns
        -------
        np.ndarray
            The first derivative of the halo density with respect to radial distance.
        """
        s = rad / self.r_s
        density = self.halo_const / (s**self.cusp * (1 + s) ** (3 - self.cusp))
        density_prime = -density / self.r_s * (3 * s + self.cusp) / s / (1 + s)
        return density_prime * eerfc(
            rad, self.chalo, self.dr_trunc, self.truncation_region_width
        ) + density * eerfcprime(
            rad, self.chalo, self.dr_trunc, self.truncation_region_width
        )

    def density_2prime(self, rad: np.ndarray) -> np.ndarray:
        """
        Computes the second derivative of the halo density with respect to radial distance.

        Parameters
        ----------
        rad : np.ndarray
            The radial distances at which to compute the second derivative of the halo density.

        Returns
        -------
        np.ndarray
            The second derivative of the halo density with respect to radial distance.
        """
        s = rad / self.r_s
        density = self.halo_const / (s**self.cusp * (1 + s) ** (3 - self.cusp))
        density_prime = -density / self.r_s * (3 * s + self.cusp) / s / (1 + s)
        density_2prime = (
            density
            / (self.r_s**2)
            * (self.cusp * (self.cusp + 1) + 8 * self.cusp * s + 12 * s**2)
            / s**2
            / (1 + s) ** 2
        )
        return (
            density_2prime
            * eerfc(rad, self.chalo, self.dr_trunc, self.truncation_region_width)
            + 2
            * density_prime
            * eerfcprime(rad, self.chalo, self.dr_trunc, self.truncation_region_width)
            + density
            * eerfc2prime(rad, self.chalo, self.dr_trunc, self.truncation_region_width)
        )

    def halo_potential_estimate(self, nr: int, dr: float):
        print("Halo Pot Estimate")

        # Initialize master radial grid
        self.dens = np.zeros(nr + 1)
        self.dr = dr
        self.nr = nr

        # Plot density from analytic function
        for ir in range(self.nr + 1):
            r = ir * self.dr
            if r == 0.0:
                r = self.eps
            self.dens[ir] = self.density(r)

        # Now get the potential harmonics of this new density (BT 2-208)
        # Using Simpson's rule integration
        self.pot, self.fr = simpson_harm_int(self.nr, self.dr, self.dens)
