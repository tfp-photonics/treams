.. highlight:: console

===============
Getting started
===============

Installation
============

To install the package with pip, use ::

   pip install treams


How to use treams
=================

Import *treams*, create T-matrices and start calculating.

.. doctest::

   >>> import treams
   >>> tm = treams.TMatrix.sphere(1, .6, 1, [4, 1])
   >>> f"{tm.xs_ext_avg:.4f}"
   '0.3072'

More detailed examples are given in :doc:`intro`.
