import argparse
import json

from galaxy_model import GalaxyParameters
from initializer import GalactICSInitializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    galaxy_params = GalaxyParameters(config)
    galaxy_params.set_defaults()

    print("Halo Parameters:", galaxy_params.halo)
    print("Disk 1 Parameters:", galaxy_params.disk1)
    print("Disk 2 Parameters:", galaxy_params.disk2)
    print("Gas Parameters:", galaxy_params.gas)
    print("Bulge Parameters:", galaxy_params.bulge)
    print("Blackhole Parameters:", galaxy_params.bh)
    print("Grid Parameters:", galaxy_params.grid)
    print("Psi Parameters:", galaxy_params.psi)

    initializer = GalactICSInitializer(galaxy_params)
    initializer.initialize()
