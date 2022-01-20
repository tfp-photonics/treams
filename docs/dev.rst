============================
Development and contributing
============================

Setting up the development environment
======================================

1. Clone the repository with

.. code-block:: console

   git clone git@git.scc.kit.edu/photonics/ptsa.git

or

.. code-block:: console

   git clone https://git.scc.kit.edu/photonics/ptsa.git

and enter the directory

.. code-block:: console

   cd ptsa

2. Create a conda environment with

.. code-block:: console

   conda env create -f environment.yml

which installs all packages needed for building and running ptsa, testing, different
benchmarks, and building the documentation. Activate the environment:

.. code-block:: console

   conda activate ptsa-dev

3. Setup the package with

.. code-block:: console

   pip install -e .

This last step makes the program available in the environment independently of the
current folder. This is especially necessary for correctly building the documentation.


Running tests
=============

Tests can be run with

.. code-block:: console

   python -m pytest

but for development a more fine grained selection can be made by passing a directory or
file as an argument. Additionally, the option ``-k`` allows to define keywords when
selecting test.

Some tests in the module :ref:`ptsa-lattice` take a long time to finish and are
therefore disabled by default. You can add them with the option ``--runslow``.

If coverage reports should be included on can use the option ``--cov ptsa``. However,
this will only report on the pure python files. To also get coverage reports for the
cython part, it is necessary to compile it with linetracing support. This can be
achieved by setting the environment variable ``CYTHON_COVERAGE``, for example with

.. code-block:: console

    CYTHON_COVERAGE=1 pip install -e .

Make sure that new C code files are generated and that those files are compiled.

Building the documentation
==========================

After setting up the development environment run

.. code-block:: console

   sphinx-build -b html docs docs/_build/html

from the root directory of the package to build the documentation as html pages.

Building the code on Windows
============================

The main issue with using ptsa on Windows is the compilation step. For Windows Python is
usually compiled with MSVC for Visual Studio. However, especially for calculations with
complex numbers, Cython creates code that conforms to the (C99-) standard.Thus, it is
not compatible with the non-standard implementation of complex numbers by Microsoft.

As I understand, even large projects like numpy and scipy do not compile their code with
MSVC, at least not completely) for the Windows distribution.

Below you find three tested ways, how one can use ptsa with Windows. The first two ways
actually use a non-Windows version of Python, but have a more straightforward
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
ptsa. First, install MSYS2 and update it according to the instructions. Then, also
install python and, if you want, the dependencies of ptsa. Otherwise, the dependencies
are installed by pip.

Compilation with mingw-w64 for MSVC Python
------------------------------------------

This is approach is different from the others, since it finally combines binaries from
two different compilers. Although it works and was tested on some systems, it is not
guaranteed that it will work for all systems. The following part describes, how ptsa can
be built for Windows. It was initially inspired by
`<https://docs.scipy.org/doc/scipy/reference/building/windows.html>`_. However, it is
not completely tested, which steps could possible be omitted.

The first step is the installation of MSYS2 and components of Microsoft Visual Studio
The installation of MSYS2 is pretty straightforward. Regarding the Microsoft Visual
Studio components, it is unclear to me, which one are actually used, so for this part we
just rely on the description for scipy, which is sufficient set. Feel free to test this
and adjust here accordingly. I suspect, that it might not be necessary to install most
components at all. The only requirement so far seems to be the presence of
`vcruntime140.dll`, which should come shipped with recent versions of Python
(see also
`Steve Dower's blog post <https://stevedower.id.au/blog/building-for-python-3-5-part-two>`_).
If not present, they can additionally be installed with the pip package `msvc-runtime`.
Obviously, an installation of Python on Windows is necessary. This can either be pure
Python or can come with a distribution like Anaconda. In some cases, it might be
necessary to patch distutils' `cygwinccompiler.py` to return `vcruntime140` instead of
`msvcr140`.

Within MSYS2 install `mingw-w64-x86_64-gcc`.

The compilation is steered from the command line. First go into the directory of ptsa.
Then, set up your path by prepending the direction for MSYS2's mingw64 binaries with
`set PATH=C:\msys2\mingw64\bin;%PATH%` (adjust accordingly if you have installed MSYS2
with non-default parameters). Check that gcc from MSYS2 is recognized correctly but
make sure that the version of python that is found on the path corresponds to the
Windows Python. With this setup building binaries should work with `python -m build`.

Other remarks
=============


.. todolist::
