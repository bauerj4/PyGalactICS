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
| [`nfw_halo_walkthrough.ipynb`](nfw_halo_walkthrough.ipynb) | **GalactICS → ntropy** end-to-end: what GalactICS does, `nfw_halo_model_fast`, `dbh` + `genhalo`, and why each diagnostic plot is made |

Artifacts (plots, particle files) land in `notebooks/artifacts/nfw_walkthrough/`.
