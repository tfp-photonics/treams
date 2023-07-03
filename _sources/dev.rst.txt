.. highlight:: console

============================
Development and contributing
============================

Setting up the development environment
======================================

1.
Clone the repository with ::

   git clone git@github.com:tfp-photonics/treams.git

or ::

   git clone https://github.com/tfp-photonics/treams.git

and enter the directory ::

   cd treams

2.
This step is optional, but you might want to create an environment where treams and the
development tools are created ::

   conda env create --name treams-dev python

Activate the environment with::

   conda activate treams-dev

3.
Setup the package with ::

   pip install -e .

This last step makes the program available in the environment independently of the
current folder. This is especially necessary for correctly building the documentation.

Running tests
=============

To install the required packages for testing use ::

   pip install treams[test]

Tests can be run using pytest with ::

   python -m pytest

but for development a more fine grained selection can be made by passing a directory or
file as an argument. Additionally, the option ``-k`` allows to define keywords when
selecting test.

Some integration tests for the module :ref:`generated/treams.lattice:treams.lattice`
take a long time to finish and are therefore disabled by default. You can add them with
the option ``--runslow``.

If coverage reports should be included one can use the option ``--cov treams`` (which
requires pytest-cov to be installed). However, this will only report on the pure python
files. To also get coverage reports for the cython part, it is necessary to compile it
with linetracing support. This can be achieved by setting the environment variable
``CYTHON_COVERAGE``, for example with ::

    CYTHON_COVERAGE=1 pip install -e .

Make sure that new C code files are generated and that those files are compiled.
Enabling the tracing for getting code coverage slows down the integration tests
considerably. For the coverage calculation only the unit tests are used.

Some test for importing and exporting need gmsh. These tests can be activated with
``--rungmsh`` analogous to the option ``--runslow`` above.

Building the documentation
==========================


To install the required packages for testing use ::

   pip install treams[docs]

The documentation is built with sphinx by using ::

   sphinx-build -b html docs docs/_build/html

from the root directory of the package to build the documentation as html pages.
Some figures are automatically created, which requires matplotlib.

The doctests can be run with ::

   sphinx-build -b doctest docs docs/_build/doctest

Building the code on Windows
============================

The main issue with using treams on Windows is the compilation step. For Windows Python
is usually compiled with MSVC for Visual Studio. However, especially for calculations
with complex numbers, Cython creates code that conforms to the (C99-) standard. Thus, it
is not compatible with the non-standard implementation of complex numbers by Microsoft.

Below you find three tested ways, how one can use treams with Windows. The first two
ways actually use a non-Windows version of Python, but have a more straightforward
installation procedure. However, the last one works with Windows' version of Python e.g.
when installed with conda under Windows.

Windows Subsystem for Linux
---------------------------

The
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install>`_
exists on recent versions of Windows. To install it, open the command line interpreter
(cmd.exe) and type `wsl --install`. This step might be enough to be able to run a Linux
kernel. Within WSL all instructions from the rest of the description can be used, e.g.
with the distribution's Python or a conda-installed Python.

Sometimes it is necessary to use
`additional steps <https://docs.microsoft.com/en-us/windows/wsl/install-manual>`_ to
install WSL.

Pure MSYS2
----------

Using `MSYS2 <https://www.msys2.org/>`_, it is also possible to compile and install
treams. First, install MSYS2 and update it according to the instructions. Then, also
install python and, if you want, the dependencies of treams. Otherwise, the dependencies
are installed by pip.

Compilation with mingw-w64
--------------------------

This is approach is different from the others, since it finally combines binaries from
two different compilers. Although it works and was tested on some systems, it is not
guaranteed that it will work for all systems. The following part describes, how treams
can be built for Windows.

After installing MSYS2 use it to install ``mingw-w64-x86_64-gcc``.

The compilation is steered from the command line. First go into the directory of treams.
Then, set up your path by prepending the direction for MSYS2's mingw64 binaries with
``set PATH=C:\msys64\mingw64\bin;%PATH%`` (adjust accordingly if you have installed
MSYS2 with non-default parameters). Check that gcc from MSYS2 is recognized correctly
but make sure that the version of python that is found on the path corresponds to the
Windows Python. With this setup, building binaries should work with `python -m build`.

Continuous integration
======================

Using Github actions the steps above (building the documentation, running the doctests,
running the tests with a coverage report, and building for all platforms) are
implemented to run automatically on specific triggers. The relevant files in the
``.gihub/workflows`` directory can be also used as examples in addition to the
description above.
