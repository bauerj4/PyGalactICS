import math
import numpy as np

from scipy.interpolate import RegularGridInterpolator, interp1d
from galaxy_model import GalaxyParameters
from disk_density import DiskDensObj
from halo_density import HaloDensObj
from bulge_density import BulgeDensObj
from gas_density import GasDensObj
from grid_potential_utils import pot_from_grid


class GalactICSInitializer:
    """
    Creates the density / potential grids for a GalactICS galaxy.
    """

    def __init__(self, config: GalaxyParameters):

        self.config = config
        self.constants = {}

    def initialize(self):
        self.initialize_galaxy_components()

    def initialize_galaxy_components(self):
        print(
            "Initializing Galaxy Components",
            self.config.disk_flag1,
            self.config.disk_flag2,
            self.config.gas_flag,
            self.config.bulge_flag,
            self.config.halo_flag,
        )

        if self.config.halo_flag:
            self.initialize_halo()

        if self.config.disk_flag1:
            self.initialize_disk()

        if self.config.disk_flag2:
            self.initialize_disk()

        if self.config.gas_flag:
            self.initialize_gas()

        if self.config.bulge_flag:
            self.initialize_bulge()

        self.initialize_potential_arrays()

        # Get sphericalized Psi for potential iteration
        self.get_total_psi_arr()

        # Populate the potential and force arrays
        self.pot_calc_ini()

        # Iterate until the system potential is represented
        # by Legendre Polynomials

        self.pot_iterate()

        # Compute distribution function
        self.initialize_distribution_function(
            self.config.psi["npsi"], self.config.psi["nint"]
        )

    def initialize_halo(self):
        """
        Initialize halo density using the parameters from GalaxyParameters.
        """
        self.halo_obj = HaloDensObj(**self.config.halo)
        self.halo_obj.halo_potential_estimate(
            self.config.grid["nr"], self.config.grid["dr"]
        )

    def initialize_disk(self):
        """
        Initialize the DiskDensObj based on parameters from GalaxyParameters
        and set up interpolation for the bulge density profile.
        """
        self.disk_obj = DiskDensObj(**self.config.disk1)
        self.disk_obj.disk_potential_estimate(
            self.config.grid["nr"], self.config.grid["dr"]
        )

    def initialize_gas(self):
        """
        Initialize the GasDensObj based on parameters from GalaxyParameters
        and set up interpolation for the bulge density profile.
        """
        self.gas_obj = GasDensObj(**self.config.gas)
        # TODO:         call getzgasgrid_old()
        self.gas_obj.gas_potential_estimate(
            self.config.grid["nr"], self.config.grid["dr"]
        )

    def initialize_bulge(self):
        """
        Initialize the BulgeDensObj based on parameters from GalaxyParameters
        and set up interpolation for the bulge density profile.
        """
        self.bulge_obj = BulgeDensObj(**self.config.bulge)

    def initialize_potential_arrays(self):
        """
        Iniitialize arrays for Legendre summation
        """
        self.pot = np.zeros((self.config.grid["lmax"], self.config.grid["nr"] + 1))
        self.fr = np.zeros((self.config.grid["lmax"], self.config.grid["nr"] + 1))

    def initialize_distribution_function(self, npsi, nint):
        # Placeholder for actual DF initialization logic
        pass

    def get_total_psi_arr(self):
        """
        Compute the total potential at radius `rad` from various galaxy components.

        Returns
        -------
        float
            The total potential at radius `rad`.
        """
        total_psi_arr = np.zeros(self.config.grid["nr"] + 1)
        total_force_arr = np.zeros(self.config.grid["nr"] + 1)
        rad = np.array([i * self.config.grid["dr"] for i in range(len(total_psi_arr))])

        # Thin disk
        if self.config.disk_flag1:
            total_psi_arr += self.disk_obj.pot
            total_force_arr += self.disk_obj.fr

        # Thick disk
        if self.config.disk_flag2:
            total_psi_arr += self.thick_disk_obj.pot
            total_force_arr += self.thick_disk_obj.fr

        # Gas disk
        if self.config.gas_flag:
            total_psi_arr += self.gas_obj.pot
            total_force_arr += self.gas_obj.fr

        # Bulge
        if self.config.bulge_flag:
            total_psi_arr += np.array([self.bulge_obj.sersic_potential(r) for r in rad])
            total_force_arr += np.array([self.bulge_obj.sersic_force(r) for r in rad])

        # Halo
        if self.config.halo_flag:
            total_psi_arr += self.halo_obj.pot
            total_force_arr += self.halo_obj.fr

        self.total_psi_arr = total_psi_arr
        self.total_force_arr = total_force_arr
        self.get_total_psi = interp1d(rad, total_psi_arr, kind="linear")
        self.get_total_force = interp1d(rad, total_force_arr, kind="linear")

    def pot_calc_ini(self):
        """
        Initialize the potential arrays. The initialization procedure
        takes the sphericalized psi we calculated before, the sphericalized radial
        forces, and a grid of shape (lmax, nr) of Legendre polynomial degrees and
        radial shells. For l = 0, we have a gravitational monopole. This will
        be populated by our spherical approximation to the potential.
        """

        print("Potential Initialize")
        self.pot[0, 0] = self.get_total_psi(0.0) * np.sqrt(4.0 * np.pi)
        for i in range(1, self.config.grid["nr"] + 1):
            r = i * self.config.grid["dr"]
            self.pot[0, i] = np.sqrt(4.0 * np.pi) * self.get_total_psi(r)
            self.fr[0, i] = -np.sqrt(4.0 * np.pi) * self.get_total_force(r)
            for l in range(2, self.config.grid["lmax"] + 1, 2):
                self.pot[l // 2, i] = 0
        print(
            "Initial Potential Check",
            self.pot[0, 1],
            self.pot[0, self.config.grid["nr"]],
        )

    def pot_iterate(self):
        print("Iteratively Solving for Potential")
