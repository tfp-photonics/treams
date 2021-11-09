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

   python setup.py develop

4. And finally install the current directory with conda

.. code-block:: console

   conda develop .

This last step makes the program available in the environment independently of the
current folder. This is especially necessary for correctly building the documentation.


Running tests
=============

Run tests with

.. code-block:: console

   python -m pytest tests

.. todo:: coverage, slow tests, ...

Building the documentation
==========================

After setting up the development environment run

.. code-block:: console

   sphinx-build -b html docs docs/_build/html

from the root directory of the package to build the documentation as html pages.

Other remarks
=============


.. todolist::
