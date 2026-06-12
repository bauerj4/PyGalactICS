#!/usr/bin/env bash
# Build legacy binaries without GNU make (gfortran + gcc only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/legacy/fortran"
BIN="$ROOT/legacy/bin"
mkdir -p "$BIN"
cd "$SRC"

F77=gfortran
CC=gcc
FFLAGS="-ffixed-line-length-0 -O -fno-backslash"
CFLAGS="-O -DRINGASCII -DASCII"
FLIBS="-lm"

echo "Compiling Fortran objects..."
for f in dbh.f omekap.f polardens.f bulgepotential.f totdens.f halopotential.f pot.f \
  diskdens.f dens.f appdiskpot.f plgndr1.f bulgedenspsi.f halodenspsi.f \
  gendenspsihalo.f gendenspsibulge.f polarbulgedens.f polarhalodens.f \
  appdiskdens.f halodens.f dfhalo.f dfbulge.f erfcen.f modstamp.f force.f \
  appdiskforce.f gendf.f getpsi.f dpolardens.f diskpotentialestimate.f \
  diskpotentialestimate2.f gaspotentialestimate.f halopotentialestimate.f \
  nfwprofiles.f sersicprofiles.f getzgas.f getgasnorm.f splined.f splintd.f; do
  $F77 $FFLAGS -c "$f"
done

echo "Linking dbh..."
$F77 $FFLAGS $FLIBS dbh.o omekap.o polardens.o bulgepotential.o totdens.o halopotential.o \
  pot.o diskdens.o dens.o appdiskpot.o plgndr1.o bulgedenspsi.o halodenspsi.o \
  gendenspsihalo.o gendenspsibulge.o polarbulgedens.o polarhalodens.o appdiskdens.o \
  halodens.o dfhalo.o dfbulge.o erfcen.o modstamp.o force.o appdiskforce.o gendf.o \
  getpsi.o dpolardens.o diskpotentialestimate.o diskpotentialestimate2.o gaspotentialestimate.o \
  halopotentialestimate.o nfwprofiles.o sersicprofiles.o getzgas.o getgasnorm.o splined.o \
  splintd.o -o "$BIN/dbh"

echo "dbh -> $BIN/dbh"
