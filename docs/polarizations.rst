============================
Polarizations and mode types
============================

Polarizations
=============

Two polarization types are supported: `helicity` and `parity`. The first allows the use
of chiral material parameters. Each polarization type contains two polarizations that
are indicated by the integers `0` and `1` throughout the code. For helicity
polarizations `0` stands for negative helicity and `1` for positive helicity. In the
case of parity polarizations `0` stands for `TE` or `magnetic` polarization and `1` for
`TM` or `electric` polarizations. The magnetic parity waves are defined in
:func:`ptsa.special.vsw_M`, :func:`ptsa.special.vsw_rM`, :func:`ptsa.special.vcw_M`,
:func:`ptsa.special.vcw_rM`, and :func:`ptsa.special.vpw_M`. For spherical waves they
are transverse with respect to the radial direction, for cylindrical and plane waves
they are transverse to the z-axis. The corresponding electric parity waves are
:func:`ptsa.special.vsw_N`, :func:`ptsa.special.vsw_rN`, :func:`ptsa.special.vcw_N`,
:func:`ptsa.special.vcw_rN`, and :func:`ptsa.special.vpw_N`.

The helicity waves are defined in :func:`ptsa.special.vsw_A`,
:func:`ptsa.special.vsw_rA`, :func:`ptsa.special.vcw_A`, :func:`ptsa.special.vcw_rA`,
and :func:`ptsa.special.vpw_A`.

The default polarization type to be used can be defined in :mod:`ptsa.config`.

Mode types
==========

For some basis sets there exist two different types of modes, that distinguish
propagation features. For the spherical and cylindrical basis theses are `regular`
and `singular` modes. The former come through the use of (spherical) Bessel Functions
and the latter through the use of (spherical) Hankel functions of the first kind. The
regular modes are finite in the whole space. Thus, they are suitable for describing
incident modes or to expand a plane wave. The singular modes fulfil the radiation
condition and as such are used for the scattered fields.

For the, here, so-called partial plane wave basis (:class:`~ptsa.PlaneWaveBasisPartial`)
only two components of the wave vector are given and the third component is only
implicitly defined by the wave number and the material parameters. The application for
this basis is mostly within stratified media that are uniform or periodic in the two
other dimensions. Thus, the two given components of the wave vectors are conserved up
to reciprocal lattice vectors. To lift the ambiguity of the definition of the third
component, the mode types `up` and `down` are possible. They define, if the modes
propagate -- or decay for evanescent modes -- along the positive or negative direction
with respect to the third axis.
