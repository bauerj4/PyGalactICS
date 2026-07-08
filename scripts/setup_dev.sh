#!/usr/bin/env bash
# Create a local venv and install galacticsics with dev dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3-venv is required. On Debian/Ubuntu: sudo apt install python3-venv"
  exit 1
fi

bash scripts/ensure_openmpi.sh || true
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install -e "src/ntropy[dev]"
if ldconfig -p 2>/dev/null | grep -q 'libmpi\.so'; then
  .venv/bin/pip install --force-reinstall --no-cache-dir mpi4py \
    || echo "WARNING: mpi4py reinstall failed. Install OpenMPI dev headers (make install-system-mpi or sudo apt install openmpi-bin libopenmpi-dev) then: pip install --force-reinstall mpi4py"
else
  echo "NOTE: OpenMPI not detected; skipping mpi4py rebuild (MPI tests need make install-system-mpi)"
fi
.venv/bin/python -c "from mpi4py import MPI; print('mpi4py OK (COMM_WORLD size =', MPI.COMM_WORLD.Get_size(), ')')" \
  || echo "WARNING: mpi4py import failed. Install OpenMPI (make install-system-mpi or sudo apt install openmpi-bin libopenmpi-dev) then: pip install --force-reinstall mpi4py"
echo "Done. Activate with: source .venv/bin/activate"
echo "Run tests: pytest tests/ -v"
