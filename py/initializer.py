import math
import numpy as np

from scipy.interpolate import RegularGridInterpolator, interp1d
from galaxy_model import GalaxyParameters
from disk_density import DiskDensObj
from halo_density import HaloDensObj
from bulge_density import BulgeDensObj
from gas_density import GasDensObj


class GalactICSInitializer:
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

        self.initialize_potential_arrays(
            self.config.grid["nr"], self.config.grid["lmax"]
        )
        self.initialize_legendre_constants(self.config.grid["lmax"])
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
        self.bulge = BulgeDensObj(**self.config.bulge)

    def initialize_potential_arrays(self, nr, lmax):
        self.constants["TotPot"] = np.zeros((nr, lmax + 1))
        self.constants["HaloPot"] = np.zeros((nr, lmax + 1))
        self.constants["BulgePot"] = np.zeros((nr, lmax + 1))

    def initialize_legendre_constants(self, lmax):
        self.constants["plcon"] = np.zeros(lmax + 1)
        for i in range(lmax + 1):
            self.constants["plcon"][i] = math.sqrt((2 * i + 1) / (4.0 * math.pi))

    def initialize_distribution_function(self, npsi, nint):
        # Placeholder for actual DF initialization logic
        pass
