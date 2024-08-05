import numpy as np
import truncation

from scipy.special import erfc
from harmonic_integration import simpson_harm_int


class DiskDensObj:
    """
    Class representing the properties of a disk.
    """

    def __init__(
        self,
        outdisk: float = 30,
        mdisk: float = 15.0,
        drtrunc: float = 0.5,
        rdisk: float = 3.5,
        rhole: float = 0.0,
        rcore: float = 3.0,
        zdisk: float = 0.25,
        truncation_region_width: float = 4.0,
        ntheta: int = 100,
        eps: float = 0.0001,
        **kwargs,
    ):
        """
        Initialize the DiskObj with the given parameters.

        Parameters
        ----------
        outdisk : float
            The outer radius of the disk (kpc). Also the centerpoint
            for truncation.
        mdisk : float
            The mass of the disk
        drtrunc : float
            The truncation width of the disk (kpc). This is the width
            of a radial ring or a bin in height.
        rdisk : float
            The scale length of the disk.
        rhole : float
            The core radius of the hole in the disk.
        rcore : float
            The core radius.
        zdisk : float
            The scale height of the disk.
        truncation_region_width : float
            The width over which to truncate.
        n_theta : int
            The number of angular bins.
        eps : float
            Small radius for density calculation (kpc)
        """
        self.outdisk = outdisk
        self.drtrunc = drtrunc
        self.rdisk = rdisk
        self.mdisk = mdisk
        self.rhole = rhole
        self.rcore = rcore
        self.zdisk = zdisk
        self.truncation_region_width = truncation_region_width
        self.ntheta = ntheta
        self.eps = eps

    @property
    def diskconst(self):
        """
        Scale factor for disk density
        """
        return self.mdisk / (2.0 * np.pi * self.rdisk**2)

    def diskvertdens(self, z):
        """
        Compute the vertical density components for a given z coordinate.

        Parameters
        ----------
        z : float
            The vertical coordinate.

        Returns
        -------
        g : float
            The vertical density function.
        g1 : float
            The first derivative of the vertical density function.
        g2 : float
            The second derivative of the vertical density function.
        """
        zz = z / self.zdisk
        if abs(zz) > 50.0:
            g = abs(zz)
            g1 = np.sign(zz)
            g2 = 0.0
        else:
            g = np.log(np.cosh(zz))
            g1 = np.tanh(zz)
            g2 = 1.0 / np.cosh(zz) ** 2

        return g, g1, g2

    def disksurfdens(self, r):
        """
        Compute the surface density and its derivatives for a given radius.

        Parameters
        ----------
        r : float
            The radial coordinate.

        Returns
        -------
        f : float
            The surface density.
        f1r : float
            The first derivative of the surface density with respect to r.
        f2 : float
            The second derivative of the surface density with respect to r.
        """
        t = np.sqrt(0.5) * (r - self.outdisk) / self.drtrunc
        t2 = t * t

        eerfc = truncation.eerfc(
            r, self.outdisk, self.drtrunc, self.truncation_region_width
        )
        eexp = truncation.eexp(
            r, self.outdisk, self.drtrunc, self.truncation_region_width
        )

        arg1 = -r / self.rdisk

        # rhole = 0.0 corresponds to no hole cropped out of the center
        if self.rhole == 0.0:
            # Surface density without hole
            sg = self.diskconst * np.exp(arg1)

            # First derivative of surface density without hole
            sg1 = -self.diskconst / self.rdisk * np.exp(arg1)

            # Second derivative of surface density without hole
            sg2 = self.diskconst / self.rdisk**2 * np.exp(arg1)
        else:
            tmp2 = np.sqrt(r**2 + self.rhole**2)
            arg2 = -tmp2 / self.rcore

            # Surface density with hole
            sg = self.diskconst * (np.exp(arg1) - np.exp(arg2))

            # First derivative of surface density with hole
            sg1 = self.diskconst * (
                -1.0 / self.rdisk * np.exp(arg1) + r / self.rcore / tmp2 * np.exp(arg2)
            )

            # Second derivative of surface density with hole
            sg2 = self.diskconst * (
                (1.0 / self.rdisk**2) * np.exp(arg1)
                + np.exp(arg2)
                * (self.rcore * self.rhole**2 - r**2 * tmp2)
                / self.rcore**2
                / tmp2**3
            )

        # Final truncated density
        f = sg * eerfc

        # Density derivatives at 0 are 0
        if r > 0:
            #  This is just the derivative with truncation
            # d sg / dr * eerfc + d eerfc / dr  * sg
            f1r = (sg1 * eerfc + eexp * sg) / r

            # Similarly, the second derivative.
            f2 = (
                sg2 * eerfc
                + 2.0 * sg1 * eexp
                + eexp * ((r - self.outdisk) / self.drtrunc**2) * sg
            )
        else:
            f1r = 0.0
            f2 = 0.0

        if np.isnan(f) or f > 1e10:
            print(
                "inside disksurf fac1, eerfc, r, rdisk, diskconst",
                f,
                eerfc,
                r,
                self.rdisk,
                self.diskconst,
            )
            raise ValueError("Density calculation error")

        return f, f1r, f2

    def diskdensestimate(self, s, z):
        """
        Estimate the disk density at a given position using surface and vertical densities.

        Parameters
        ----------
        s : float
            The radial distance from the center.
        z : float
            The vertical coordinate.

        Returns
        -------
        density_estimate : float
            The estimated disk density.
        """
        r = np.sqrt(s**2 + z**2)
        f, f1r, f2 = self.disksurfdens(r)
        g, g1, g2 = self.diskvertdens(z)

        density_estimate = 0.5 * f * g2 / self.zdisk

        return density_estimate

    def dpolardiskdens(self, r, ctheta):
        """
        Compute the disk density in polar coordinates.

        Parameters
        ----------
        r : float
            The radial coordinate.
        ctheta : float
            The cosine of the polar angle.

        Returns
        -------
        density : float
            The disk density at the given polar coordinates.
        """
        z = r * ctheta
        s = r * np.sqrt(1.0 - ctheta**2)
        return self.diskdensestimate(s, z)

    def disk_potential_estimate(self, nr: int, dr: float):
        print("Disk Pot Estimate")

        # Initialize master radial grid
        self.pot = np.zeros(nr + 1)
        self.fr = np.zeros(nr + 1)
        self.dens = np.zeros(nr + 1)
        self.dr = dr
        self.nr = nr

        # Density
        s = 0.0

        for ir in range(nr + 1):

            r = ir * self.dr
            if r == 0.0:
                r = self.eps

            cthetamax = min(1.0, 10.0 * self.zdisk / r)
            dctheta = cthetamax / float(self.nr)

            s = self.dpolardiskdens(r, cthetamax) + self.dpolardiskdens(r, 0.0)

            for is_ in range(1, self.ntheta - 1, 2):
                ctheta = float(is_) * dctheta
                s += 4 * self.dpolardiskdens(r, ctheta)

            for is_ in range(2, self.ntheta - 2, 2):
                ctheta = float(is_) * dctheta
                s += 2 * self.dpolardiskdens(r, ctheta)

            s = s * dctheta / 3.0
            self.dens[ir] = s

        # Now get the potential harmonics of this new density (BT 2-208)
        # Using Simpson's rule integration
        self.pot, self.fr = simpson_harm_int(self.nr, self.dr, self.dens)

    def appdiskpot(self, s: float, z: float) -> float:
        """
        Compute the approximate disk potential at a point (s, z).

        Parameters
        ----------
        s : float
            Radial distance in the disk plane.
        z : float
            Vertical distance from the disk plane.
        disk_obj : DiskObj
            Disk object containing attributes like `zdisk`.

        Returns
        -------
        float
            Approximate disk potential.
        """
        r = np.sqrt(s**2 + z**2)

        f, f1r, f2 = self.disksurfdens(r)

        if f == 0.0:
            return 0.0
        else:
            g, g1, g2 = self.diskvertdens(z)
            potential = -4 * np.pi * f * self.zdisk * g / 2.0
            return potential
