===============
Getting started
===============

Installation
============

Installation using conda
------------------------

Currently ptsa is not available as a conda package. A local repository can be installed
in development mode (see :ref:`dev:Setting up the development environment`).

Installation using pip
----------------------

To install the package with pip, use

.. code-block:: console

   pip install git+https://git.scc.kit.edu/photonics/ptsa.git


Manual local installation
-------------------------

This is not recommended.

Either clone the repository via git or download and unzip the files. Then change into
the new directory and execute

.. code-block:: console

   python setup.py build_ext --inplace

You can use the package only from its own directory.


How to use ptsa
===============

.. todo::

   Give a very short example on how to use the code


Where to go from here
=====================

A couple of simulations with varying degree of complexity can be found in
:doc:`examples`. There's also a complete :doc:`ptsa` of the API. If you want to get
contribute to the development of the program, you can find relevant information under
:doc:`dev`.
