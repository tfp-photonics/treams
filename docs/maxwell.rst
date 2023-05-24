=====================================================
Maxwell's equations and chiral constitutive relations
=====================================================

In matter Maxwell's equations can be written in frequency domain in the absence of free
charges and currents as

.. math::

    \nabla \cdot
    \begin{pmatrix}
        \frac{1}{\epsilon_0} \boldsymbol D \\
        c \boldsymbol B
    \end{pmatrix}
    = 0

and

.. math::

    \nabla \times
    \begin{pmatrix}
        \boldsymbol E \\
        Z_0 \boldsymbol H
    \end{pmatrix}
    =
    k_0
    \begin{pmatrix}
        0 & \mathrm i \\
        - \mathrm i & 0
    \end{pmatrix}
    \begin{pmatrix}
        \frac{1}{\epsilon_0} \boldsymbol D \\
        c \boldsymbol B
    \end{pmatrix}

where :math:`\boldsymbol E`, :math:`\boldsymbol H`, :math:`\boldsymbol D`, and
:math:`\boldsymbol B` are the electric and magnetic fields, the displacement field, and
the magnetic flux density (:func:`treams.efield`, :func:`treams.hfield`,
:func:`treams.dfield`, :func:`treams.bfield`). All these quantities are complex valued
fields, that depend on the angular frequency :math:`\omega` and the position
:math:`\boldsymbol r`, which we omitted here for a conciser notation. The speed of light
(in vacuum) :math:`c`, the free space impedance :math:`Z_0`, and the vacuum permittivity
:math:`\epsilon_0` are chosen as constant prefactors such that all fields are normalized
to the same units. Conventionally, within treams the (vacuum) wave number
:math:`k_0 = \frac{\omega}{c}` is generally used to express the frequency.

For the transformation to the time domain we use for a general function
:math:`f(\omega)`

.. math::

    f(t) = \int_{-\infty}^\infty \mathrm d t f(\omega) \mathrm e^{-\mathrm i \omega t}

as Fourier transformation convention, and thus the inverse transformation is

.. math::

    f(\omega)
    = \int_{-\infty}^\infty \frac{\mathrm d \omega}{2 \pi}
    f(t) \mathrm e^{\mathrm i \omega t}


To solve those equations they have to be complemented with constitutive relations. In a
linear, time-invariant, homogeneous, isotropic, local and reciprocal medium the relation
of the four electromagnetic fields can be expressed by

.. math::

    \begin{pmatrix}
        \frac{1}{\epsilon_0} \boldsymbol D \\
        c \boldsymbol B
    \end{pmatrix}
    =
    \begin{pmatrix}
        \epsilon & \mathrm i \kappa \\
        - \mathrm i \kappa & \mu
    \end{pmatrix}
    \begin{pmatrix}
        \boldsymbol E \\
        Z_0 \boldsymbol H
    \end{pmatrix}

where :math:`\epsilon`, :math:`\mu`, and :math:`\kappa` are the relative permittivity,
relative permeability, and chirality parameter (:class:`treams.Material`). Due to the
requirement of isotropy these quantities are all scalar.

The combination of the curl equation and the constitutive relations leads to the
equation

.. math::

    \nabla \times
    \begin{pmatrix}
        \boldsymbol E \\
        Z_0 \boldsymbol H
    \end{pmatrix}
    =
    k_0
    \begin{pmatrix}
        \kappa & \mathrm i \mu \\
        - \mathrm i \epsilon & \kappa
    \end{pmatrix}
    \begin{pmatrix}
        \boldsymbol E \\
        Z_0 \boldsymbol H
    \end{pmatrix}

that can be diagonalized to yield

.. math::

    \nabla \times
    \begin{pmatrix}
        \boldsymbol G_- \\
        \boldsymbol G_+
    \end{pmatrix}
    =
    \begin{pmatrix}
        -k_- & 0 \\
        0 & k_+
    \end{pmatrix}
    \begin{pmatrix}
        \boldsymbol G_- \\
        \boldsymbol G_+
    \end{pmatrix}

where the Riemann-Silberstein vectors :math:`\sqrt{2} \boldsymbol G_\pm = \boldsymbol E
\pm \mathrm i Z_0 Z \boldsymbol H` appear (:func:`treams.gfield`), with the relative
impedance defined as :math:`Z = \sqrt{\frac{\mu}{\epsilon}}`. The wave numbers in the
medium are :math:`k_\pm = k_0 n_pm = k_0 (n \pm \kappa)` with the refractive index
:math:`n = \sqrt{\epsilon \mu}`.

The alternative definition of the Riemann-Silberstein vectors :math:`\sqrt{2}
\boldsymbol F_\pm = \frac{1}{\epsilon_0 \epsilon} \boldsymbol D \pm \mathrm i
\frac{c}{n} \boldsymbol B` using the displacement field and the magnetic flux density
instead of the electric and magnetic field is related to the definition above by
:math:`\boldsymbol F_\pm = \frac{n_\pm}{n} \boldsymbol G_\pm` (:func:`treams.ffield`).

In isotropic media the divergence equations simply become :math:`\nabla
\boldsymbol G_\pm = 0 = \nabla \boldsymbol F_\pm`.

Solutions to the vector Helmholtz equation
==========================================

Instead of immediatly solving Maxwell's equations from above, we will study the
Helmholtz equation which is commonly encountered when studying wave phenomena first.
This section mainly relies on [#]_.

The vector Helmholtz equation is

.. math::

    (\Delta + k^2) \boldsymbol f
    =
    \nabla (\nabla \boldsymbol f)
    - \nabla \times \nabla \times \boldsymbol f
    + k^2 \boldsymbol f
    = 0

where :math:`\Delta` is the Laplace operator. Note, that by applying the curl operator
twice on the Riemann-Silberstein vectors (in the case of an achiral material this is
also true for the electric and magnetic fields) and using the transversality condition
for the fields, the vector Helmholtz equation can be easily obtained.

Solutions to the vector Helmholtz equation can be obtained from solutions to the scalar
Helmholtz equation :math:`(\Delta + k^2) f = \nabla (\nabla f) - k^2 f = 0` by using the
construction

.. math::

    \boldsymbol L = \boldsymbol v f \\
    \boldsymbol M = \nabla \times (\boldsymbol v f) \\
    \boldsymbol N = \nabla \times \nabla \times (\boldsymbol v f)

where :math:`\boldsymbol v` is a steering vector that depends on the coordinate system
used for the solution :math:`f`. We will focus the following discussion on the three
cases of planar, cylindrical, and spherical solutions, where the coordinate systems are
chosen to be Cartesian, cylindrical, and spherical. Also, we will limit the discussion
of the first type of solution, because it is not transverse.

Plane waves
-----------

In Cartesian coordinates the solution to the scalar Helmholtz equation are simple
plane waves :math:`\mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}` where the wave
vector fulfils :math:`\boldsymbol k^2 = k_x^2 + k_y^2 + k_z^2 = k^2`. The steering
vector is constant and conventionally chosen to be the unit vector along the z-axis
:math:`\boldsymbol{\hat z}`. Then, the solutions

.. math::

    \boldsymbol M_{\boldsymbol k} (k, \boldsymbol r)
    =
    \mathrm i
    \frac{k_y \boldsymbol{\hat x} - k_x \boldsymbol{\hat y}}{\sqrt{k_x^2 + k_y^2}}
    \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}
    = 
    -\mathrm i \boldsymbol{\hat \varphi}_{\boldsymbol k}
    \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}
    \\
    \boldsymbol N_{\boldsymbol k} (k, \boldsymbol r)
    =
    \frac{-k_x k_z \boldsymbol{\hat x} - k_y k_z \boldsymbol{\hat y} + (k_x^2 + k_y^2)
    \boldsymbol{\hat z}}{k\sqrt{k_x^2 + k_y^2}}
    \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}
    = -\boldsymbol{\hat \theta}_{\boldsymbol k}
    \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}

are found (:func:`treams.special.vpw_M` and :func:`treams.special.vpw_N`). We normalized
these solutions such that they have unit strength for real-valued wave vectors. The
solution :math:`\boldsymbol M_{\boldsymbol k}` is always perpendicular to the z-axis.
Thus, with respect to the x-y-plane those solutions are often referred to as `TE`, when
taken for the electric field. Similarly, the solutions
:math:`\boldsymbol M_{\boldsymbol k}` are referred to as `TM`.

Cylindrical waves
-----------------

The cylindrical solutions can be constructed mostly analogously to the plane waves. The
steering vector stays :math:`\boldsymbol{\hat z}`. The solutions in cylindrical
coordinates are :math:`Z_m^{(n)}(k_\rho \rho) \mathrm e^{\mathrm i (m \varphi + k_z z}`
where :math:`k_z \in \mathbb R` and :math:`m \in \mathbb Z` are the parameters of the
solution. The radial part of the wave vector is defined as :math:`k_\rho =
\sqrt{k^2 - k_z^2}` with the imaginary part of the square root to be taken non-negative.
The functions :math:`Z_m^{(n)}` are the Bessel and Hankel functions. For a complete set
of solutions it is necessary to select two of them. We generally use the (regular)
Bessel functions :math:`J_m = Z_m^{(1)}` and the Hankel functions of the first kind
:math:`H_m^{(1)} = Z_m^{(3)}` which are singular and correspond to radiating waves
(:func:`treams.special.jv`, :func:`treams.special.hankel1`). So, the cylindrical wave
solutions are

.. math::

    \boldsymbol M_{k_z, m}^{(n)} (k, \boldsymbol r)
    =
    \left(\frac{\mathrm i m}{k_\rho \rho} Z_m^{(n)}(k_\rho \rho) \boldsymbol{\hat \rho}
    - {Z_m^{(n)}}'(k_\rho \rho) \boldsymbol{\hat \varphi}\right)
    \mathrm e^{\mathrm i (m \varphi + k_z z)}
    \\
    \boldsymbol N_{k_z, m}^{(n)} (k, \boldsymbol r)
    =
    \left(\frac{\mathrm i k_z}{k} {Z_m^{(1)}}'(k_\rho \rho) \boldsymbol{\hat \rho}
    - \frac{m k_z}{k k_\rho \rho} Z_m^{(1)}(k_\rho \rho) \boldsymbol{\hat \varphi}
    + \frac{k_\rho}{k} Z_m^{(1)}(k_\rho \rho) \boldsymbol{\hat z}\right)
    \mathrm e^{\mathrm i (k_z z + m \varphi)}

where we, again, normalized the functions (:func:`treams.special.vcw_rM`,
:func:`treams.special.vcw_M`, :func:`treams.special.vcw_rN`, and
:func:`treams.special.vcw_N`). Since the steering vector is in the direction of the
z-axis, the solutions :math:`\boldsymbol M_{k_z, m}^{(n)}` lie always in the x-y-plane.

Spherical waves
---------------

Finally, we define the spherical solutions starting from the scalar solutions
:math:`z_l^{(n)}(kr) Y_{lm}(\theta, \phi)` where :math:`z_l^{(n)}` are the spherical
Bessel and Hankel functions (and we choose :math:`j_l = z_l^{(1)}` and
:math:`h_l^{(1)} = z_l^{(n)}` in complete analogy to the cylindrical case) and
:math:`Y_{lm}` are the spherical harmonics (:func:`treams.special.spherical_jn`,
:func:`treams.special.spherical_hankel1`, and :func:`treams.special.sph_harm`). The
value :math:`l \in \mathbb N` refers to the angular momentum. The value :math:`l = 0` is
only possible for longitudinal modes. So, for electromagnetic waves generally
:math:`l \geq 1`. The projection of the angular momentum onto the z-axis is :math:`m \in
\mathbb Z` with :math:`|m| \leq l`. The steering vector for the spherical coordinate
solution is :math:`\boldsymbol r`. Then, the vector spherical waves are defined as

.. math::

    \boldsymbol M_{lm}^{(n)} (k, \boldsymbol r)
    = z_l^{(n)} (kr) \boldsymbol X_{lm}(\theta, \varphi)
    \\
    \boldsymbol N_{lm}^{(n)} (k, \boldsymbol r)
    =
    \left({h_l^{(1)}}'(kr) + \frac{h_l^{(1)}(kr)}{kr}\right)
    \boldsymbol Y_{lm}(\theta, \varphi)
    + \sqrt{l (l + 1)} \frac{h_l^{(1)}(kr)}{kr} \boldsymbol Z_{lm}(\theta, \varphi)

(:func:`treams.special.vsw_rM`, :func:`treams.special.vsw_M`,
:func:`treams.special.vsw_rN`, and :func:`treams.special.vsw_N`) where

.. math::

    \boldsymbol X_{lm} (\theta, \varphi)
    = \mathrm i \sqrt{\frac{2 l + 1}{4 \pi l (l + 1)} \frac{(l - m)!}{(l + m)!}}
    \left(\mathrm i \pi_l^m(\cos\theta) \boldsymbol{\hat\theta}
    - \tau_l^m (\cos\theta) \boldsymbol{\hat\varphi}\right)
    \mathrm e^{\mathrm i m \varphi}
    \\
    \boldsymbol Y_{lm} (\theta, \varphi)
    = \mathrm i \sqrt{\frac{2 l + 1}{4 \pi l (l + 1)} \frac{(l - m)!}{(l + m)!}}
    \left(\tau_l^m (\cos\theta) \boldsymbol{\hat\theta}
    + \mathrm i \pi_l^m (\cos\theta) \boldsymbol{\hat\varphi}\right)
    \mathrm e^{\mathrm i m \varphi}
    \\
    \boldsymbol Z_{lm} (\theta, \varphi)
    = \mathrm i Y_{lm}(\theta, \varphi) \boldsymbol{\hat r}

are the vector spherical harmonics (:func:`treams.special.vsh_X`,
:func:`treams.special.vsh_Y`, and :func:`treams.special.vsh_Z`). These are themselves
defined by the functions :math:`\pi_l^m(x) = \frac{m P_l^m(x)}{\sqrt{1 - x^2}}`,
:math:`\tau_l^m(x) = \frac{\mathrm d}{\mathrm d \theta}P_l^m(x = \cos\theta)`, and
the associated Legendre polynomials :math:`P_l^m` (:func:`treams.special.pi_fun`,
:func:`treams.special.tau_fun`, and :func:`treams.special.lpmv`). The vector spherical
harmonics are orthogonal to each other and normalized to 1 upon integration over the
solid angle.

The solutions :math:`\boldsymbol M_{lm}^{(n)}` are transverse to a sphere due to the
steering vector pointing in the radial direction. They are referred to as `TE` but
-- confusingly -- also as `magnetic` because they correspond to the electric field of a
magnetic multipole. Conversely, the solutions :math:`\boldsymbol N_{lm}^{(n)}` are
called `TM` or `electric`.

Solutions to Maxwell's equations
================================

Up to now, we set up Maxwell's equations together with constitutive relations for chiral
media and found solutions to the vector Helmholtz equation. Next, we want to combine
those results.

Modes of well-defined helicity
------------------------------

First, we want to find solutions to the Riemann-Silberstein vectors
:math:`\boldsymbol G_\pm`. Although we can obtain the vector Helmholtz equation from
:math:`\nabla \times \boldsymbol G_\pm = \pm k_\pm \boldsymbol G_\pm`, we observe that
this equation is more restrictive, namely our solutions :math:`\boldsymbol M_\nu` and
:math:`\boldsymbol N_\nu`, where :math:`\nu` is just a placeholder for the actual
parameters that indexes the concrete set of solutions, are no solutions for it. However,
with the above definitions we find that :math:`\nabla \times
\boldsymbol M_\nu (k, \boldsymbol r) = k \boldsymbol N_\nu (k, \boldsymbol r)` and
:math:`\nabla \times \boldsymbol N_\nu(k, \boldsymbol r) = k \boldsymbol M_\nu
(k, \boldsymbol r)`. So, the combinations :math:`\sqrt{2} \boldsymbol A_{\pm,\nu}
(k, \boldsymbol r) = \boldsymbol N_\nu (k, \boldsymbol r) \pm \boldsymbol M_\nu
(k, \boldsymbol r)` are indeed solutions for the respective Riemann-Silberstein vectors
(:func:`treams.special.vpw_A`, :func:`treams.special.vcw_rA`,
:func:`treams.special.vcw_A`, :func:`treams.special.vsw_rA`, and
:func:`treams.special.vsw_A`). The solution for Maxwell's equations are then

.. math::

    \boldsymbol G_\pm(\boldsymbol r)
    = \sqrt{2} \sum_\nu a_{\pm,\nu} \boldsymbol A_{\pm,\nu} (k_\pm, \boldsymbol r)
    \\
    \boldsymbol F_\pm(\boldsymbol r)
    =
    \sqrt{2} \frac{n_\pm}{n}
    \sum_\nu a_{\pm,\nu}\boldsymbol A_{\pm,\nu} (k_\pm, \boldsymbol r)
    \\
    \boldsymbol E(\boldsymbol r)
    = \sum_{s,\nu} a_{s,\nu} \boldsymbol A_{s,\nu} (k_s, \boldsymbol r)
    \\
    Z_0 \boldsymbol H(\boldsymbol r)
    = -\frac{\mathrm i}{Z}
    \sum_{s,\nu} s a_{s,\nu} \boldsymbol A_{s,\nu} (k_s, \boldsymbol r)
    \\
    \frac{1}{\epsilon_0} \boldsymbol D(\boldsymbol r)
    = \frac{1}{Z} \sum_{s,\nu} n_s a_{s,\nu} \boldsymbol A_{s,\nu} (k_s, \boldsymbol r)
    \\
    c \boldsymbol B(\boldsymbol r)
    = -\mathrm i \sum_{s,\nu} s n_s a_{s,\nu} \boldsymbol A_{s,\nu} (k_s, \boldsymbol r)

and, because each of the individual modes is an eigenmode of the helicity operator
:math:`\frac{\nabla\times}{k}`, we call them `helicity` modes. Modes of well-defined
helicity are suitable solutions chiral media.

Parity modes
------------

When considering only achiral media, it is quite common to not use modes of well-defined
helicity but modes with well-defined parity, which are exactly the modes
:math:`\boldsymbol M_\nu` and :math:`\boldsymbol N_\nu` defined above. For achiral
materials, we have :math:`k_\pm = k` and by substituting :math:`\sqrt{2} a_{\pm,\nu} =
a_{N,\nu} \pm a_{M,\nu}` for the expansion coefficients we find the solutions

.. math::

    \boldsymbol E(\boldsymbol r)
    = \sum_\nu (a_{M,\nu} \boldsymbol M_\nu (k, \boldsymbol r)
    + a_{N,\nu} \boldsymbol N_\nu (k, \boldsymbol r))
    \\
    Z_0 \boldsymbol H(\boldsymbol r)
    = -\frac{\mathrm i}{Z}
    \sum_\nu (a_{N,\nu} \boldsymbol M_\nu (k, \boldsymbol r)
    + a_{M,\nu} \boldsymbol N_\nu (k, \boldsymbol r))
    \\
    \frac{1}{\epsilon_0} \boldsymbol D(\boldsymbol r)
    = \epsilon \sum_\nu (a_{M,\nu} \boldsymbol M_\nu (k, \boldsymbol r)
    + a_{N,\nu} \boldsymbol N_\nu (k, \boldsymbol r))
    \\
    c \boldsymbol B(\boldsymbol r)
    = -\mathrm i n \sum_\nu (a_{N,\nu} \boldsymbol M_\nu (k, \boldsymbol r)
    + a_{M,\nu} \boldsymbol N_\nu (k, \boldsymbol r))

for the parity modes.

References
==========

.. [#] P. M. Morse and H. Feshbach, Methods of Theoretical Physics
   (McGraw-Hill, New York, 1953).
