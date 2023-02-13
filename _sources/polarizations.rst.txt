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