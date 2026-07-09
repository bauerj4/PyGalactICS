# Notebooks

Tutorial notebooks for the PyGalactICS / ntropy rewrite. Outputs are written to
`notebooks/artifacts/` (gitignored).

## Setup

```bash
make install-dev
pip install jupyter ipykernel
jupyter notebook notebooks/
```

## Contents

| Notebook | Description |
|----------|-------------|
| [`nfw_halo_walkthrough.ipynb`](nfw_halo_walkthrough.ipynb) | **GalactICS → ntropy** end-to-end: IC generation, force accuracy, parallelism scaling, **|ΔE/E₀|** symplectic (leapfrog) vs explicit (Euler, RK2–4) energy drift, density stability |

Artifacts (plots, particle files) land in `notebooks/artifacts/nfw_walkthrough/`.
