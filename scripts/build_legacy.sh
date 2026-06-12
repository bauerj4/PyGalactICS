#!/usr/bin/env bash
# Build legacy GalactICS Fortran/C binaries into legacy/bin/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/legacy/fortran"

if ! command -v gfortran >/dev/null; then
  echo "gfortran is required (sudo apt install gfortran)"
  exit 1
fi
if ! command -v gcc >/dev/null; then
  echo "gcc is required"
  exit 1
fi
if ! command -v make >/dev/null; then
  echo "make is required (sudo apt install make)"
  exit 1
fi

make all
make install
echo "Legacy binaries installed to $ROOT/legacy/bin/"
