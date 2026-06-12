galacticsics
============

Python library for GalactICS galaxy potential solving, distribution-function
correction, particle sampling, and observational fitting.

Units
-----

- Length: kpc
- Velocity: 100 km/s
- Mass: 2.325 × 10⁹ M☉
- G = 1

Quick start
-----------

.. code-block:: python

   from galacticsics.builder import GalaxyBuilder
   from galacticsics.models import GalaxyModel
   from galacticsics.potential import evaluate_potential

   builder = GalaxyBuilder(
       model=GalaxyModel.milky_way_disk_halo(),
       model_dir="models/MilkyWay",
   ).load_artifacts()
   psi = evaluate_potential(builder.potential, 8.0, 0.0)

API reference
-------------

.. automodule:: galacticsics
   :members:
