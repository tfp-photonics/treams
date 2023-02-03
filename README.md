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
The code is documented with docstrings and `sphinx`. Run
`sphinx-build -b html docs docs/_build/html` and you'll find a nice HTML-page at
`docs/_build/index.html`. For this to work properly, install the program using the
development environment method.

## Features

* [x] T-matrix calculations using a spherical or cylindrical wave basis set
* [x] Calculations in helicity and parity (TE/TM) basis
* [x] Scattering from clusters of particles
* [x] Scattering from particles and clusters arranged in 3d-, 2d-, and 1d-lattices
* [x] Calculation of light propagation in stratified media
* [x] Band calculation in crystal structures
