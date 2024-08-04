import json
import argparse


class GalaxyParameters:
    """ """

    def __init__(self, config):
        self.halo = config["halo"]
        self.disk1 = config["disk1"]
        self.disk2 = config["disk2"]
        self.gas = config["gas"]
        self.bulge = config["bulge"]
        self.bh = config["bh"]
        self.grid = config["grid"]
        self.psi = config["psi"]

    @property
    def halo_flag(self):
        return self.halo["include"]

    @property
    def disk_flag1(self):
        return self.disk1["include"]

    @property
    def disk_flag2(self):
        return self.disk2["include"]

    @property
    def gas_flag(self):
        return self.gas["include"]

    @property
    def bulge_flag(self):
        return self.bulge["include"]

    @property
    def bh_flag(self):
        return self.bh["include"]

    def set_defaults(self):
        if not self.halo_flag:
            self.halo["chalo"] = 0.0
            self.halo["v0"] = 0.0
            self.halo["a"] = 0.0
            self.halo["dr_trunc_halo"] = 0.0
            self.halo["cusp"] = 0.0

        if not self.disk_flag1:
            self.disk1["disk_use_flag"] = False
            self.disk1["rm_disk"] = 0.0

        if not self.disk_flag2:
            self.disk2["disk_use_flag"] = False
            self.disk2["rm_disk"] = 0.0

        if not self.gas_flag:
            self.gas["rm_gas"] = 0.0
            self.gas["r_gas"] = 0.0
            self.gas["out_gas"] = 0.0
            self.gas["z_gas0"] = 0.0
            self.gas["dr_trunc_gas"] = 0.0
            self.gas["gamma"] = 0.0

        if not self.bulge_flag:
            self.bulge["nnn"] = 0.0
            self.bulge["ppp"] = 0.0
            self.bulge["v0bulge"] = 0.0
            self.bulge["abulge"] = 0.0

        if not self.bh_flag:
            self.bh["bh_mass"] = 0.0
