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
| [`gpu_bh_dbh.ipynb`](gpu_bh_dbh.ipynb) | **GPU Barnes–Hut DBH**: 5M disk + 1M bulge + 1M halo, single adaptive `tiered_leapfrog` + particle dumps |
| [`nfw_halo_walkthrough.ipynb`](nfw_halo_walkthrough.ipynb) | GalactICS → ntropy end-to-end NFW halo walkthrough |
| [`campaign_density_walkthrough.ipynb`](campaign_density_walkthrough.ipynb) | MW campaign density evolution & disk projections |

GPU BH implementation notes: [`src/ntropy/ntropy/forces/GPU_BLACKWELL.md`](../src/ntropy/ntropy/forces/GPU_BLACKWELL.md).

Artifacts for the GPU DBH notebook land in `notebooks/artifacts/gpu_bh_dbh/`.
