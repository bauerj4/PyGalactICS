#!/usr/bin/env bash
# Build gendisk, genhalo, genbulge, getfreqs, diskdf into legacy/bin/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/legacy/fortran"
BIN="$ROOT/legacy/bin"
cd "$SRC"

F77=gfortran
CC=gcc
FFLAGS="-ffixed-line-length-0 -O -fno-backslash"
CFLAGS="-O -DRINGASCII -DASCII"
FLIBS="-lm"

fortran_common=(
  appdiskforce.f force.f readharmfile.f pot.f plgndr1.f sersicprofiles.f
  nfwprofiles.f getzgas.f appdiskdens.f dpolardens.f splined.f splintd.f
  getpsi.f erfcen.f modstamp.f
)

for f in "${fortran_common[@]}"; do
  $F77 $FFLAGS -c "$f"
done

# diskdf / gendisk chain
for f in diskdf.f diskdf5intez.f diskdf5ez.f diskdf3intez.f diskdf3ez.f diskdensf.f \
  sigr2.f sigz2.f omekap.f diskdens.f fnamidden.f rcirc.f readdiskdf.f \
  appdiskpot.f gendisk.c genhalo.c genbulge.c; do
  case "$f" in
    *.c) $CC $CFLAGS -c "$f" ;;
    *) $F77 $FFLAGS -c "$f" ;;
  esac
done

$CC $CFLAGS -c query.c simpson.c golden.c readmassrad.c
$F77 $FFLAGS -c ran3.f

$F77 $FFLAGS $FLIBS diskdf.o appdiskpot.o diskdf5intez.o diskdf3intez.o plgndr1.o \
  diskdensf.o splintd.o splined.o readharmfile.o sigr2.o sigz2.o fnamidden.o \
  rcirc.o diskdens.o omekap.o pot.o modstamp.o getzgas.o appdiskdens.o \
  appdiskforce.o dpolardens.o nfwprofiles.o sersicprofiles.o force.o \
  -o "$BIN/diskdf"

$F77 $CFLAGS gendisk.o query.o ran3.o golden.o simpson.o readdiskdf.o diskdf5ez.o \
  diskdensf.o readharmfile.o sigr2.o sigz2.o omekap.o diskdens.o splined.o \
  splintd.o diskdf3intez.o diskdf3ez.o pot.o fnamidden.o appdiskpot.o plgndr1.o \
  rcirc.o getpsi.o sersicprofiles.o getzgas.o appdiskdens.o appdiskforce.o \
  dpolardens.o nfwprofiles.o force.o -o "$BIN/gendisk"

$F77 $FFLAGS -c halodens.f dfhalo.f dfbulge.f halodenspsi.f bulgedenspsi.f readdenspsi.f
$F77 $FFLAGS -c getfreqs.f

$F77 $FFLAGS $FLIBS getfreqs.o pot.o appdiskpot.o plgndr1.o erfcen.o getzgas.o \
  appdiskdens.o appdiskforce.o dpolardens.o nfwprofiles.o sersicprofiles.o \
  -o "$BIN/getfreqs"

$F77 $CFLAGS genhalo.o readmassrad.o ran3.o query.o readharmfile.o pot.o halodens.o \
  dfhalo.o dfbulge.o appdiskpot.o plgndr1.o halodenspsi.o readdenspsi.o erfcen.o \
  getpsi.o sersicprofiles.o force.o appdiskforce.o appdiskdens.o getzgas.o \
  dpolardens.o nfwprofiles.o -o "$BIN/genhalo"

$F77 $CFLAGS genbulge.o readmassrad.o ran3.o query.o readharmfile.o pot.o dfbulge.o \
  dfhalo.o appdiskpot.o plgndr1.o bulgedenspsi.o readdenspsi.o erfcen.o getpsi.o \
  sersicprofiles.o force.o appdiskforce.o getzgas.o appdiskdens.o dpolardens.o \
  nfwprofiles.o -o "$BIN/genbulge"

echo "Installed: diskdf gendisk getfreqs genhalo genbulge -> $BIN/"
