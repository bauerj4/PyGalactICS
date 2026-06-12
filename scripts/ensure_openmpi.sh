#!/usr/bin/env bash
# Ensure system OpenMPI libraries are available for mpi4py.
set -euo pipefail

libmpi_present() {
  ldconfig -p 2>/dev/null | grep -q 'libmpi\.so' && command -v mpirun >/dev/null
}

if libmpi_present; then
  exit 0
fi

echo "OpenMPI not detected — mpi4py needs libmpi.so and mpirun."

install_debian() {
  local pkgs="openmpi-bin libopenmpi-dev"
  if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    echo "Installing ${pkgs}..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ${pkgs}
    return 0
  fi
  if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y ${pkgs}
    return 0
  fi
  echo "Install OpenMPI manually, then rerun make install-dev:"
  echo "  sudo apt-get update && sudo apt-get install -y ${pkgs}"
  return 1
}

if command -v apt-get >/dev/null; then
  install_debian || exit 0
elif command -v dnf >/dev/null; then
  if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo dnf install -y openmpi openmpi-devel
  else
    echo "Install OpenMPI: sudo dnf install -y openmpi openmpi-devel"
    exit 0
  fi
elif command -v brew >/dev/null; then
  brew install open-mpi
else
  echo "Install OpenMPI or MPICH, then: pip install --force-reinstall mpi4py"
  exit 0
fi

if libmpi_present; then
  echo "OpenMPI ready: $(mpirun --version 2>/dev/null | head -1)"
else
  echo "OpenMPI packages installed but libmpi not yet visible; you may need a new shell."
fi
