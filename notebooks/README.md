# Notebooks

Usage guides for the PyGalactICS / ntropy rewrite. Outputs go to
`notebooks/artifacts/` (gitignored).

Related docs: [`ic_sampling.md`](../docs/ic_sampling.md) · [`ci_essential.md`](../docs/ci_essential.md) · [`galacticsics_pipeline.md`](../docs/galacticsics_pipeline.md)

## Setup

```bash
make install-dev
pip install jupyter ipykernel
python setup.py build_ext --inplace   # OpenMP samplers (_sampler_c)
# Restart Kernel after building extensions
jupyter notebook notebooks/
```

## Contents

| Notebook | Description |
|----------|-------------|
| [`gpu_bh_dbh.ipynb`](gpu_bh_dbh.ipynb) | **GPU Barnes–Hut DBH**: disk+bulge+halo ICs → OpenMP sample → adaptive `tiered_leapfrog` + `gpu_bh`; IC potential virial |
| [`nfw_halo_walkthrough.ipynb`](nfw_halo_walkthrough.ipynb) | Halo-only: Python `dbh` → OpenMP `genhalo` → BH accuracy / scaling / stability |
| [`campaign_density_walkthrough.ipynb`](campaign_density_walkthrough.ipynb) | MW campaign: configure → solve → OpenMP sample → evolve → ρ(r) / disk projections |

GPU BH implementation notes: [`src/ntropy/ntropy/forces/GPU_BLACKWELL.md`](../src/ntropy/ntropy/forces/GPU_BLACKWELL.md).

PR regression gate: `make test-essential`.
