import numpy as np
import truncation

from scipy.interpolate import interp1d, RectBivariateSpline
from harmonic_integration import simpson_harm_int


class GasDensObj:
    def __init__(
        self,
        rmgas: float = None,
        rgas: float = None,
        outgas: float = None,
        zgas0: float = None,
        drtruncgas: float = None,
        gamma: float = None,
        gasconst: float = None,
        ntheta: int = 100,
        nrsplg: int = None,
        nrmass: int = None,
        drgas: float = None,
        truncation_region_width: float = 4.0,
        eps: float = 0.0001,
        **kwargs,
    ):
        """
        Initialize the GasDensObj class for the gas disk in GalactICS.

        Parameters
        ----------
        rmgas : float
            Gas mass.
        rgas : float
            Radial scale length.
        outgas : float
            Outer truncation radius.
        zgas0 : float
            Vertical scale height.
        drtruncgas : float
            Radial truncation width.
        gamma : float
            Power-law index for density profile.
        gasconst : float
            Gas constant.
        ntheta : int
            Number of angular grid points.
        nrsplg : int
            Number of spline grid points for potential calculation.
        nrmass : int
            Number of radial mass points.
        drgas : float
            Radial grid spacing.
        truncation_region_width : float
            The truncation region width
        eps : float
            Small radius for density calculation (kpc)
        """

        self.rmgas = rmgas
        self.rgas = rgas
        self.outgas = outgas
        self.zgas0 = zgas0
        self.drtruncgas = drtruncgas
        self.gamma = gamma
        self.gasconst = gasconst
        self.truncation_region_width = truncation_region_width

        self.nrsplg = nrsplg
        self.eps = eps

        # Allocate normalization arrays
        self.GasNorm = np.zeros(nrsplg)
        self.GasNormRad = np.zeros(nrsplg)
        self.GasNorm2 = np.zeros(nrsplg)
        self.dGasNorm = np.zeros(nrsplg)
        self.dGasNorm2 = np.zeros(nrsplg)

        self.nrmass = nrmass
        self.drgas = drgas

        # initialize grid
        # self._setup_grid()

    @property
    def rgasmax(self):
        return self.outgas + self.truncation_region_width

    def _setup_grid(self, nr: int):
        """
        Set up the interpolation grid based on the given parameters.

        This method defines the radial and vertical grids and sets up
        the cubic spline interpolation for the density values on this grid.

        Attributes
        ----------
        r_grid : ndarray
            Radial grid points.
        z_grid : ndarray
            Vertical grid points.
        density_values : ndarray
            Density values at the grid points.
        density_spline : RectBivariateSpline
            Cubic spline interpolation of the density values.
        """
        # Define the grid parameters
        self.r_grid = np.linspace(0, self.rgasmax, nr)
        self.z_grid = np.linspace(0, self.zgas0 * 2, nr)

        # Example density values for the grid (replace with actual data or method)
        self.density_values = np.zeros((len(self.r_grid), len(self.z_grid)))

        # Example: Fill the density_values with some meaningful data
        # This should be replaced with actual density computation
        for i, r in enumerate(self.r_grid):
            for j, z in enumerate(self.z_grid):
                # TODO: compute density
                pass

        # Set up the cubic spline interpolation
        self.density_spline = RectBivariateSpline(
            self.r_grid, self.z_grid, self.density_values
        )

    def get_density(self, r: float, z: float) -> float:
        """
        Interpolate the density at a given (r, z) position using the cubic spline.

        Parameters
        ----------
        r : float
            Radial coordinate.
        z : float
            Vertical coordinate.

        Returns
        -------
        float
            Interpolated density value.
        """
        return self.density_spline(r, z)[0, 0]

    def gasdensestimate(self, s, z):
        """
        Estimate the gas density based on the given s and z coordinates.

        Parameters
        ----------
        s : float
            The radial distance in the plane of the disk.
        z : float
            The vertical distance from the plane of the disk.

        Returns
        -------
        float
            The estimated gas density at coordinates (s, z).
        """
        r = np.sqrt(s**2 + z**2)
        f, f1r, f2 = self.gassurfdens(r)
        zgas = self.getzgas(r)
        g, g1, g2 = self.gasvertdens(z, zgas)

        return 0.5 * f * g2 / zgas

    def gassurfdens(self, r):
        """
        Calculate the gas surface density and its derivatives at radius r.

        Parameters
        ----------
        r : float
            Radial distance.

        Returns
        -------
        tuple
            Surface density (f), its first derivative with respect to r (f1r),
            and its second derivative with respect to r (f2).
        """
        eerfc = truncation.eerfc(
            r, self.outgas, self.drtrunc, self.truncation_region_width
        )
        eexp = truncation.eexp(
            r, self.outgas, self.drtrunc, self.truncation_region_width
        )

        if r > 0:
            fac1 = self.gasconst * np.exp(-r / self.rgas)
            f = fac1 * eerfc
            f1r = -fac1 * (eerfc / self.rgas + eexp) / r
            f2 = fac1 * (
                eerfc / (self.rgas**2)
                + eexp * (2 / self.rgas + (r - self.outgas) / self.dr_trunc**2)
            )
        else:
            fac1 = self.gasconst
            f = fac1 * eerfc
            f1r = 0.0
            f2 = 0.0

        # Check for NaN or excessively large values
        if np.isnan(f) or f > 1e10:
            print(
                f"Warning: Surface density calculation resulted in NaN or excessively large value."
            )
            print(
                f"Parameters: fac1={fac1}, eerfc={eerfc}, r={r}, rgas={self.rgas}, gasconst={self.gasconst}"
            )
            raise ValueError("Invalid surface density calculation")

        return f, f1r, f2

    def _compute_zgas_grid(self):
        """
        Compute the grid of vertical heights (zgas) for the gas.
        """
        for ir in range(self.nr):
            r = ir * self.dr
            if r == 0:
                radj = np.finfo(float).eps
            else:
                radj = r

            if self.gamma < 0:
                rmax = self.rgasmax
                tdns = self._total_density(rmax)
                tdns *= 4.0 * np.pi
                f, f1r, f2 = self._surface_density(rmax)
                f *= 2.0 * np.pi
                zgasofrmax = (
                    -f / self.truncation_region_width
                    + np.sqrt(
                        f**2 / self.truncation_region_width**2 + self.zgas0 * tdns
                    )
                ) / tdns

                rarg = (radj - rmax) / self.dr_trunc
                if rarg > self.truncation_region_width:
                    errfac = 1.0
                    zgasofr = 0.0
                else:
                    errfac = 0.5 * (self._erf(rarg) + 1.0)
                    tdns = self._total_density(radj)
                    tdns *= 4.0 * np.pi
                    f, f1r, f2 = self._surface_density(self.outgas)
                    f *= 2.0 * np.pi

                zgasofr = (
                    -f / self.truncation_region_width
                    + np.sqrt(
                        f**2 / self.truncation_region_width**2 + self.zgas0 * tdns
                    )
                ) / tdns
            else:
                zgasofr = self.zgas0

            self.zgasgrid[ir] = (1.0 - errfac) * zgasofr + errfac * zgasofr

    def getzgas(self, r):
        """
        Get the scale height of the gas at radius r.

        Parameters
        ----------
        r : float
            Radial distance.

        Returns
        -------
        float
            The scale height of the gas at radius r.
        """
        return self.get_zgas_interp(r)

    def gasvertdens(self, z, zgas):
        """
        Calculate the vertical density and its derivatives.

        Parameters
        ----------
        z : float
            Vertical distance from the plane of the disk.
        zgas : float
            The scale height of the gas.

        Returns
        -------
        tuple
            A tuple containing the vertical density, its first derivative with respect to z,
            and its second derivative with respect to z.
        """
        zz = z / zgas
        if abs(zz) > 50.0:
            g = abs(zz)
            g1 = 1.0
            g2 = 0.0
        else:
            g = np.log(np.cosh(zz))
            g1 = np.tanh(zz)
            g2 = 1.0 / np.cosh(zz) ** 2
        return g, g1, g2

    def appgasdens(self, s, z):
        """
        Calculate the approximate gas density.

        Parameters
        ----------
        s : float
            Radial distance in the plane of the disk.
        z : float
            Vertical distance from the plane of the disk.

        Returns
        -------
        float
            The approximate gas density at coordinates (s, z).
        """
        r = np.sqrt(s**2 + z**2)
        f, f1r, f2 = self.gassurfdens(r)
        h, h1, h2 = self.gasscaleheight(s)
        g, g1, g2 = self.gasvertdens(z, h)

        if s > 0:
            appgasdens = (
                f2 * h * g
                + 2 * f1r * g * (h + h1 * s)
                + 2.0 * f1r * g1 * z * (1.0 - s * h1 / h)
                + f * g2 / h * (1.0 + (z * h1 / h) ** 2)
                - f * g1 * z / h * (h2 + h1 / s)
                + f * g * (h2 + h1 / s)
            )
        else:
            appgasdens = (
                f2 * h * g
                + 2 * f1r * g * h
                + 2.0 * f1r * g1 * z * (1.0 - s * h1 / h)
                + f * g2 / h * (1.0 + (z * h1 / h) ** 2)
                - f * g1 * z / h * h2
                + f * g * h2
            )

        return appgasdens / 2.0

    def gasscaleheight(self, r):
        """
        Calculate the gas scale height and its derivatives at radius r.

        Parameters
        ----------
        r : float
            Radial distance.

        Returns
        -------
        tuple
            Scale height (h), its first derivative with respect to r (h1), and its second derivative with respect to r (h2).
        """
        h = self.get_zgas(r)
        if r == 0:
            h1 = 0.0
            h2 = 0.0
            return h, h1, h2

        deltar = min(self.dr, r / 10.0)
        hm = self.get_zgas(r - deltar / 2.0)
        hp = self.get_zgas(r + deltar / 2.0)
        h1 = (hp - hm) / deltar
        h2 = self.truncation_region_width * (hp - 2.0 * h + hm) / (deltar**2)

        # Check for large values of h2
        # TODO: figure out where 18 comes from
        if np.abs(h2) > 18.0 * self.truncation_region_width:
            print(f"Warning: h2 value is large.")
            print(
                f"Parameters: r={r}, deltar={deltar}, h1={h1}, h2={h2}, h={h}, hm={hm}, hp={hp}"
            )
            raise ValueError("Large value for h2")

        return h, h1, h2

    def dpolardiskdens(self, r, ctheta):
        """
        Compute the gas density in polar coordinates.

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
        return self.diskdensestimate(self, s, z)

    def gas_potential_estimate(self, nr: int, dr: float):
        print("Gas Pot Estimate")

        # Potential
        s = 0.0

        # Initialize master radial grid
        self.pot = np.zeros(nr)
        self.fr = np.zeros(nr)
        self.dens = np.zeros(nr)
        self.dr = dr
        self.nr = nr

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

    def appgaspot(self, s: float, z: float) -> float:
        """
        Compute the approximate gas potential at a point (s, z).

        Parameters
        ----------
        s : float
            Radial distance in the gas plane.
        z : float
            Vertical distance from the gas plane.

        Returns
        -------
        float
            Approximate gas potential.
        """
        r = np.sqrt(s**2 + z**2)
        zgas = self.getzgas(s)

        f, f1r, f2 = self.gassurfdens(r)

        if f == 0.0:
            return 0.0
        else:
            g, g1, g2 = self.gasvertdens(z, zgas)
            potential = -4 * pi * f * zgas * g / 2.0
            return potential
