.. highlight:: console

===============
Getting started
===============

Installation
============

To install the package with pip, use ::

   pip install git+https://github.com/tfp-photonics/treams.git


How to use treams
=================

Import *treams*, create T-matrices and start calculating.

.. doctest::

   >>> import treams
   >>> tm = treams.TMatrix.sphere(1, .6, 1, [4, 1])
   >>> tm.xs_ext_avg
   0.3072497765576123

More detailed examples are given in :doc:`intro`.
