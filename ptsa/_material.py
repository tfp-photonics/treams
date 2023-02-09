import numpy as np

from ptsa import misc


class Material:
    r"""Material definition.

    The material properties are defined in the frequency domain through scalar values
    for permittivity :math:`\epsilon`, permeability :math:`\mu`, and the chirality
    parameter  :math:`\kappa`. Materials are, thus, assumed to be linear,
    time-invariant, homogeneous, isotropic, and local. Also, it is assumed that they
    have no gain. The relation of the electric and magnetic fields is defined by

    .. math::

        \begin{pmatrix}
            \frac{1}{\epsilon_0} \boldsymbol D \\
            c \boldsymbol B
        \end{pmatrix}
        =
        \begin{pmatrix}
            \epsilon & \mathrm i \kappa \\
            -\mathrm i \kappa & \mu
        \end{pmatrix}
        \begin{pmatrix}
            \boldsymbol E \\
            Z_0 \boldsymbol H
        \end{pmatrix}

    for a fixed vacuum wave number :math:`k_0` and spatial position
    :math:`\boldsymbol r` with :math:`\epsilon_0` the vacuum permittivity, :math:`c` the
    speed of light in vacuum, and :math:`Z_0` the vacuum impedance.

    Args:
        epsilon (optional, complex): Relative permittivity. Defaults to 1.
        mu (optional, complex): Relative permeability. Defaults to 1.
        kappa (optional, complex): Chirality parameter. Defaults to 0.
    """

    def __init__(self, epsilon=1, mu=1, kappa=0):
        """Initialization."""
        if isinstance(epsilon, Material):
            epsilon, mu, kappa = epsilon()
        elif isinstance(epsilon, (tuple, list, np.ndarray)):
            if len(epsilon) == 0:
                epsilon = 1
            elif len(epsilon) == 1:
                epsilon = epsilon[0]
            elif len(epsilon) == 2:
                epsilon, mu = epsilon
            elif len(epsilon) == 3:
                epsilon, mu, kappa = epsilon
            else:
                raise ValueError("invalid material definition")
        self._epsilon = epsilon
        self._mu = mu
        self._kappa = kappa

    @property
    def epsilon(self):
        """Relative permittivity.

        Returns:
            float or complex
        """
        return self._epsilon

    @property
    def mu(self):
        """Relative permeability.

        Returns:
            float or complex
        """
        return self._mu

    @property
    def kappa(self):
        """Chirality parameter.

        Returns:
            float or complex
        """
        return self._kappa

    def __iter__(self):
        """Iterator for a tuple containing the material parameters.

        Useful for unpacking the material parameters into a function that takes these
        parameters separately, e.g. ``foo(*material)``.
        """
        return iter((self.epsilon, self.mu, self.kappa))

    @classmethod
    def from_n(cls, n=1, impedance=1, kappa=0):
        r"""Create material from refractive index and relative impedance.

        This function calculates the relative permeability and permittivity with
        :math:`\epsilon = \frac{n}{Z}` and :math:`\mu = nZ`. The chirality parameter is
        considered separately.

        Note:
            The refractive index is defined independently from the chirality parameter
            here. For an alternative definition of the refractive index, see also
            :func:`Material.from_nmp`.

        Args:
            n (complex, optional): Refractive index. Defaults to 1.
            impedance(complex, optional): Relative impedance. Defaults to 1.
            kappa (complex, optional): Chirality parameter.

        Returns:
            Material
        """
        epsilon = n / impedance
        mu = n * impedance
        return cls(epsilon, mu, kappa)

    @classmethod
    def from_nmp(cls, ns=(1, 1), impedance=1):
        r"""Create material from refractive indices of both helicities.

        This function calculates the relative permeability and permittivity and the
        chirality parameter with :math:`\epsilon = \frac{n_+ + n_-}{2Z}`,
        :math:`\mu = \frac{(n_+ + n_-)Z}{2}` and :math:`\mu = \frac{(n_+ - n_-)}{2}`.

        Note:
            Two refractive indices are defined here that depend on the chirality
            parameter. For an alternative definition of the refractive index, see also
            :func:`Material.from_n`.

        Args:
            ns ((2,)-array-like, optional): Negative and positive helicity refractive
                index. Defaults to (1, 1).
            impedance(complex, optional): Relative impedance. Defaults to 1.

        Returns:
            Material
        """
        epsilon = sum(ns) * 0.5 / impedance
        mu = sum(ns) * 0.5 * impedance
        kappa = (ns[1] - ns[0]) * 0.5
        return cls(epsilon, mu, kappa)

    @property
    def n(self):
        r"""Refractive index.

        The refractive index is defined by :math:`n = \sqrt{\epsilon \mu}`, with an
        enforced non-negative imaginary part.

        Note:
            The refractive index returned is independent from the chirality parameter.
            For an alternative definition of the refractive index, see also
            :func:`Material.nmp`.

        Returns:
            complex
        """
        n = np.sqrt(self.epsilon * self.mu)
        if n.imag < 0:
            n = -n
        return n

    @property
    def nmp(self):
        r"""Refractive indices of both helicities.

        The refractive indices are defined by
        :math:`n_\pm = \sqrt{\epsilon \mu} \pm \kappa`, with an enforced non-negative
        imaginary part.

        Note:
            The refractive indices returned depend on the chirality parameter. For an
            alternative definition of the refractive index, see also :func:`Material.n`.

        Returns:
            tuple
        """
        return misc.refractive_index(self.epsilon, self.mu, self.kappa)

    @property
    def impedance(self):
        r"""Relative impedance.

        The relative impedance is defined by :math:`Z = \sqrt{\frac{\epsilon}{\mu}}`.

        Returns:
            complex
        """
        return np.sqrt(self.mu / self.epsilon)

    def __call__(self):
        """Return a tuple containing all material parameters.

        Returns:
            tuple
        """
        return self.epsilon, self.mu, self.kappa

    def __eq__(self, other):
        """Compare material parameters.

        Materials are considered equal, when all material parameters are equal. Also,
        compares with objects that contain at most three values.

        Returns:
            bool
        """
        if other is None:
            return False
        if not isinstance(other, Material):
            other = Material(*other)
        return (
            self.epsilon == other.epsilon
            and self.mu == other.mu
            and self.kappa == other.kappa
        )

    @property
    def ischiral(self):
        """Test if the material is chiral.

        Returns:
            bool
        """
        return self.kappa != 0

    @property
    def isreal(self):
        """Test if the material has purely real parameters.

        Returns:
            bool
        """
        return all(i.imag == 0 for i in self)

    def __str__(self):
        """All three material parameters.

        Returns:
            str
        """
        return "(" + ", ".join([str(i) for i in self()]) + ")"

    def __repr__(self):
        """Representation that allows recreating the object.

        Returns:
            str
        """
        return self.__class__.__name__ + str(self)

    def ks(self, k0):
        """Return the wave numbers in the medium for both polarizations.

        The first value corresponds to negative helicity and the second to positive
        helicity. For achiral materials where parity polarizations can be used both
        values are equal.

        Args:
            k0 (float): Wave number in vacuum.

        Returns:
            tuple
        """
        return k0 * self.nmp

    def krhos(self, k0, kz, pol):
        r"""The (cylindrically) radial part of the wave vector.

        The cylindrically radial part is defined by :math:`k_\rho = \sqrt(k^2 - k_z^2)`.
        In case of chiral materials :math:`k` and so :math`k_\rho` depends on the
        polarization. The returned values have non-negative imaginary parts.

        Args:
            k0 (float): Wave number in vacuum.
            kz (float, array-like): z-component of the wave vector
            pol (int, array-like): Polarization indices
                (:ref:`polarizations:Polarizations`).

        Returns:
            complex, array-like
        """
        ks = self.ks(k0)[pol]
        return misc.wave_vec_z(kz, 0, ks)

    def kzs(self, k0, kx, ky, pol):
        r"""The z-component of the wave vector.

        The z-component of the wave vector is defined by
        :math:`k_z = \sqrt(k^2 - k_x^2 - k_y^2)`. In case of chiral materials :math:`k`
        and so :math`k_z` depends on the polarization. The returned values have
        non-negative imaginary parts.

        Args:
            k0 (float): Wave number in vacuum.
            kx (float, array-like): x-component of the wave vector
            ky (float, array-like): y-component of the wave vector
            pol (int, array-like): Polarization indices
                (:ref:`polarizations:Polarizations`).

        Returns:
            complex, array-like
        """
        ks = self.ks(k0)[pol]
        return misc.wave_vec_z(kx, ky, ks)
