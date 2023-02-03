# ptsa

The package "**p**eriodic **T**-matrix **s**cattering **a**lgorithms" provides a
framework to simplify computations of the electromagnetic scattering of waves at finite
and at periodic arrangements of particles.

## Installation

### Installation using pip

To install the package with pip, use

```sh
pip install git+https://github.com/tfp-photonics/ptsa.git
```

If you're using the system wide installed version of python, you might consider the
``--user`` option.

### Running on Windows

For Windows, there are currently two tested ways how to install ptsa. The first option
is using the
[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install).
Within WSL ptsa can be installed just as described above. The second option, that was
tested is using [MSYS2](https://www.msys2.org/) with the ``mingw64`` environment.
Likely, other python versions based on ``mingw-w64`` might also work.

## Documentation

The documentation can be found at https://tfp-photonics.github.io/ptsa.

## Publications

When using this code please cite:

[D. Beutel, A. Groner, C. Rockstuhl, C. Rockstuhl, and I. Fernandez-Corbaton, Efficient Simulation of Biperiodic, Layered Structures Based on the T-Matrix Method, J. Opt. Soc. Am. B, JOSAB 38, 1782 (2021).](https://doi.org/10.1364/JOSAB.419645)

Other relevant publications are
* [D. Beutel, I. Fernandez-Corbaton, and C. Rockstuhl, Unified Lattice Sums Accommodating Multiple Sublattices for Solutions of the Helmholtz Equation in Two and Three Dimensions, Phys. Rev. A 107, 013508 (2023).](https://doi.org/10.1103/PhysRevA.107.013508)
* [D. Beutel, P. Scott, M. Wegener, C. Rockstuhl, and I. Fernandez-Corbaton, Enhancing the Optical Rotation of Chiral Molecules Using Helicity Preserving All-Dielectric Metasurfaces, Appl. Phys. Lett. 118, 221108 (2021).](https://doi.org/10.1063/5.0050411)


## Features

* [x] T-matrix calculations using a spherical or cylindrical wave basis set
* [x] Calculations in helicity and parity (TE/TM) basis
* [x] Scattering from clusters of particles
* [x] Scattering from particles and clusters arranged in 3d-, 2d-, and 1d-lattices
* [x] Calculation of light propagation in stratified media
* [x] Band calculation in crystal structures
