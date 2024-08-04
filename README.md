# GalactICS

GalactICS is an initial conditions code based on the work of Larry Widrow, John Dubinski, and Koen Kuijken. I received a copy from Nathan Deg's [web page](https://nathandeg.com/galactics/) (thanks Nathan). Once apon a time, I was Larry's Ph.D. student and remember what this code is supposed to do. My goal with this project is to document and bring it into the 21st century where Python has become the lingua Franca in science. While the code is mine, the algorithm is certainly not. Please give the attribution to the original authors in the NOTICE.md file according to the terms of Apache 2.0.

## Installation

### Set Makeflags

### Poetry Install

In a virtual environment, install the python dependencies with:
```
pip install poetry
poetry install
```

### Fortran Compiler

Ensure that you have `gfortran`. On Linux, you can run
```bash
sudo apt-get install gfortran gcc-multilib
```

### Run Makefile

Uncomment `-m32` in `GalactICS/src/Makefile` if you are actually running on a 32 bit OS... From the `GalactICS/src` directory, run:
```
make
make install
```
and the binaries will be output in the GalactICS/bin directory if everything compiles correctly.