"""Basis sets and core array functionalities."""

import abc
import copy

import numpy as np

import treams._operators as op
import treams.lattice as la
from treams import util
from treams._lattice import Lattice
from treams._material import Material


class BasisSet(util.OrderedSet, metaclass=abc.ABCMeta):
    """Basis set base class.

    It is the base class for all basis sets used. They are expected to be an ordered
    sequence of the modes, that are included in a expansion. Basis sets are expected to
    be immutable.
    """

    _names = ()
    """Names of the relevant parameters"""

    def __repr__(self):
        """String representation.

        Automatically generated when the attribute ``_names`` is defined.

        Returns:
            str
        """
        string = ",\n    ".join(f"{name}={i}" for name, i in zip(self._names, self[()]))
        return f"{self.__class__.__name__}(\n    {string},\n)"

    def __len__(self):
        """Number of modes."""
        return len(self.pol)

    @classmethod
    @abc.abstractmethod
    def default(cls, *args, **kwargs):
        """Construct a default basis from parameters.

        Construct a basis set in a default order by giving few parameters.
        """
        raise NotImplementedError


class SphericalWaveBasis(BasisSet):
    r"""Basis of spherical waves.

    Functions of the spherical wave basis are defined by their angular momentum ``l``,
    its projection onto the z-axis ``m``, and the polarization ``pol``. If the basis
    is defined with respect to a single origin it is referred to as "global", if it
    contains multiple origins it is referred to as "local". In a local basis an
    additional position index ``pidx`` is used to link the modes to one of the
    specified ``positions``.

    For spherical waves there exist multiple :ref:`polarizations:Polarizations` and they
    can be separated into incident and scattered fields. Depending on these combinations
    the basis modes refer to one of the functions :func:`~treams.special.vsw_A`,
    :func:`~treams.special.vsw_rA`, :func:`~treams.special.vsw_M`,
    :func:`~treams.special.vsw_rM`, :func:`~treams.special.vsw_N`, or
    :func:`~treams.special.vsw_rN`.

    Args:
        modes (array-like): A tuple containing a list for each of ``l``, ``m``, and
            ``pol`` or ``pidx``, ``l``, ``m``, and``pol``.
        positions (array-like, optional): The positions of the origins for the specified
            modes. Defaults to ``[[0, 0, 0]]``.

    Attributes:
        pidx (array-like): Integer referring to a row in :attr:`positions`.
        l (array-like): Angular momentum as an integer :math:`l > 0`
        m (array-like): Angular momentum projection onto the z-axis, it is an integer
            with :math:`m \leq |l|`
        pol (array-like): Polarization, see also :ref:`polarizations:Polarizations`.
    """

    _names = ("pidx", "l", "m", "pol")

    def __init__(self, modes, positions=None):
        """Initalization."""
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            pidx = []
            l = []  # noqa: E741
            m = []
            pol = []
        elif len(modes[0]) == 3:
            l, m, pol = (*zip(*modes),)
            pidx = np.zeros_like(l)
        elif len(modes[0]) == 4:
            pidx, l, m, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        if positions is None:
            positions = np.zeros((1, 3))
        positions = np.array(positions, float)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")

        self.pidx, self.l, self.m, self.pol = [
            np.array(i, int) for i in (pidx, l, m, pol)
        ]
        for i, j in ((self.pidx, pidx), (self.l, l), (self.m, m), (self.pol, pol)):
            i.flags.writeable = False
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.l < 1):
            raise ValueError("'l' must be a strictly positive integer")
        if np.any(self.l < np.abs(self.m)):
            raise ValueError("'|m|' cannot be larger than 'l'")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        if np.any(self.pidx >= len(positions)):
            raise ValueError("undefined position is indexed")

        self._positions = positions
        self._positions.flags.writeable = False
        self.hints = {}

    @property
    def positions(self):
        """Positions of the modes' origins.

        The positions are an immutable (N, 3)-array. Each row corresponds to a point in
        the three-dimensional Cartesian space.
        """
        return self._positions

    def __repr__(self):
        """String representation."""
        positions = "positions=" + str(self.positions).replace("\n", ",")
        return f"{super().__repr__()[:-1]}    {positions},\n)"

    @property
    def isglobal(self):
        """Basis is defined with respect to a single (global) origin.

        Returns:
            bool
        """
        return len(self) == 0 or np.all(self.pidx == self.pidx[0])

    def __getitem__(self, idx):
        """Get a subset of the basis.

        This function allows index into the basis set by an integer, a slice, a sequence
        of integers or bools, an ellipsis, or an empty tuple. All of them except the
        integer and the empty tuple results in another basis set being returned. In case
        of the two exceptions a tuple is returned.

        Alternatively, the string "plmp", "lmp", or "lm" can be used to access only a
        subset of :attr:`pidx`, :attr:`l`, :attr:`m`, and :attr:`pol`.
        """
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "plmp":
                return self.pidx, self.l, self.m, self.pol
            elif idx == "lmp":
                return self.l, self.m, self.pol
            elif idx == "lm":
                return self.l, self.m
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.l[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or (isinstance(idx, tuple) and len(idx) == 0):
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        """Compare basis sets.

        Basis sets are considered equal, when they have the same modes in the same order
        and the specified origin :attr:`positions` are equal.
        """
        try:
            return self is other or (
                np.array_equal(self.pidx, other.pidx)
                and np.array_equal(self.l, other.l)
                and np.array_equal(self.m, other.m)
                and np.array_equal(self.pol, other.pol)
                and np.array_equal(self.positions, other.positions)
            )
        except AttributeError:
            return False

    @classmethod
    def default(cls, lmax, nmax=1, positions=None):
        """Default basis for the given maximal multipolar order.

        The default order contains separate blocks for each position index which are in
        ascending order. Within each block the modes are sorted by angular momentum
        :math:`l`, with the lowest angular momentum coming first. For each angular
        momentum its z-projection is in ascending order from :math:`m = -l` to
        :math:`m = l`. Finally, the polarization index is the fastest changing index
        which iterates between 1 and 0.

        Example:
            >>> SphericalWaveBasis.default(2)
            SphericalWaveBasis(
                pidx=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
                l=[1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2],
                m=[-1 -1  0  0  1  1 -2 -2 -1 -1  0  0  1  1  2  2],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[0. 0. 0.]],
            )
            >>> SphericalWaveBasis.default(1, 2, [[0, 0, 1.], [0, 0, -1.]])
            SphericalWaveBasis(
                pidx=[0 0 0 0 0 0 1 1 1 1 1 1],
                l=[1 1 1 1 1 1 1 1 1 1 1 1],
                m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[ 0.  0.  1.], [ 0.  0. -1.]],
            )

        Args:
            lmax (int): Maximal multipolar order.
            nmax (int, optional): Number of positions, defaults to 1.
            positions (array-like, optional): Positions of the origins.
        """
        modes = [
            [n, l, m, p]
            for n in range(0, nmax)
            for l in range(1, lmax + 1)  # noqa: E741
            for m in range(-l, l + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, positions=positions)

    @classmethod
    def ebcm(cls, lmax, nmax=1, mmax=-1, positions=None):
        """Order of modes suited for ECBM.

        In comparison to :meth:`default` this order prioritises blocks of the
        z-projection of the angular momentum :math:`m` over the angular momentum
        :math:`l`. This is useful to get block-diagonal matrices for the extended
        boundary condition method (EBCM).

        Example:
            >>> SphericalWaveBasis.ebcm(2)
            SphericalWaveBasis(
                pidx=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
                l=[2 2 1 1 2 2 1 1 2 2 1 1 2 2 2 2],
                m=[-2 -2 -1 -1 -1 -1  0  0  0  0  1  1  1  1  2  2],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[0. 0. 0.]],
            )

        Args:
            lmax (int): Maximal multipolar order.
            nmax (int, optional): Number of particles, defaults to 1.
            mmax (int, optional): Maximal value of `|m|`. If ommitted or set to -1 it
                is taken equal to `lmax`.
            positions (array-like, optional): Positions of the origins.

        Returns:
            tuple
        """
        mmax = lmax if mmax == -1 else mmax
        modes = [
            [n, l, m, p]
            for n in range(0, nmax)
            for m in range(-mmax, mmax + 1)
            for l in range(max(abs(m), 1), lmax + 1)  # noqa: E741
            for p in range(1, -1, -1)
        ]
        return cls(modes, positions=positions)

    @staticmethod
    def defaultlmax(dim, nmax=1):
        """Calculate the default mode order for a given length.

        Given the dimension of the T-matrix return the estimated maximal value of `l`.
        This is the inverse of :meth:`defaultdim`. A value of zero is allowed for empty
        T-matrices.

        Example:
            >>> SphericalWaveBasis.defaultlmax(len(SphericalWaveBasis.default(3)))
            3

        Args:
            dim (int): Dimension of the T-matrix, respectively number of modes.
            nmax (int, optional): Number of particles, defaults to 1.

        Returns:
            int
        """
        res = np.sqrt(1 + dim * 0.5 / nmax) - 1
        res_int = int(np.rint(res))
        if np.abs(res - res_int) > 1e-8 * np.maximum(np.abs(res), np.abs(res_int)):
            raise ValueError("cannot estimate the default lmax")
        return res_int

    @staticmethod
    def defaultdim(lmax, nmax=1):
        """Default number of modes for a given mulipolar order.

        Given the maximal value of `l` return the size of the corresponding T-matrix.
        This is the inverse of :meth:`defaultlmax`. A value of zero is allowed.

        Args:
            lmax (int): Maximal multipolar order
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            int
        """
        # zero is allowed and won't give an error
        if lmax < 0 or nmax < 0:
            raise ValueError("maximal order must be positive")
        return 2 * lmax * (lmax + 2) * nmax

    def _from_iterable(cls, it, positions=None):
        if isinstance(cls, SphericalWaveBasis):
            positions = cls.positions if positions is None else positions
            cls = type(cls)
        obj = cls(it, positions=positions)
        return obj


class CylindricalWaveBasis(BasisSet):
    r"""Basis of cylindrical waves.

    Functions of the cylindrical wave basis are defined by the z-components of the wave
    vector ``kz`` and the angular momentum  ``m`` as well as the polarization ``pol``.
    If the basis is defined with respect to a single origin it is referred to as
    "global", if it contains multiple origins it is referred to as "local". In a local
    basis an additional position index ``pidx`` is used to link the modes to one of the
    specified ``positions``.

    For cylindrical waves there exist multiple :ref:`polarizations:Polarizations` and
    they can be separated into incident and scattered fields. Depending on these
    combinations the basis modes refer to one of the functions
    :func:`~treams.special.vcw_A`, :func:`~treams.special.vcw_rA`,
    :func:`~treams.special.vcw_M`, :func:`~treams.special.vcw_rM`,
    :func:`~treams.special.vcw_N`, or :func:`~treams.special.vcw_rN`.

    Args:
        modes (array-like): A tuple containing a list for each of ``kz``, ``m``, and
            ``pol`` or ``pidx``, ``kz``, ``m``, and``pol``.
        positions (array-like, optional): The positions of the origins for the specified
            modes. Defaults to ``[[0, 0, 0]]``.

    Attributes:
        pidx (array-like): Integer referring to a row in :attr:`positions`.
        kz (array-like): Real valued z-component of the wave vector.
        m (array-like): Integer angular momentum projection onto the z-axis.
        pol (array-like): Polarization, see also :ref:`polarizations:Polarizations`.
    """

    _names = ("pidx", "kz", "m", "pol")

    def __init__(self, modes, positions=None):
        """Initalization."""
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            pidx = []
            kz = []
            m = []
            pol = []
        elif len(modes[0]) == 3:
            kz, m, pol = (*zip(*modes),)
            pidx = np.zeros_like(m)
        elif len(modes[0]) == 4:
            pidx, kz, m, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        if positions is None:
            positions = np.zeros((1, 3))
        positions = np.array(positions, float)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")

        self.pidx, self.m, self.pol = [np.array(i, int) for i in (pidx, m, pol)]
        self.kz = np.array(kz, float)
        self.kz.flags.writeable = False
        for i, j in ((self.pidx, pidx), (self.m, m), (self.pol, pol)):
            i.flags.writeable = False
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        if np.any(self.pidx >= len(positions)):
            raise ValueError("undefined position is indexed")

        self._positions = positions
        self._positions.flags.writeable = False

        self.hints = {}
        if len(self.kz) > 0 and np.all(self.kz == self.kz[0]):
            self.hints["kpar"] = [np.nan, np.nan, self.kz[0]]

    @property
    def positions(self):
        """Positions of the modes' origins.

        The positions are an immutable (N, 3)-array. Each row corresponds to a point in
        the three-dimensional Cartesian space.
        """
        return self._positions

    def __repr__(self):
        """String representation."""
        positions = "positions=" + str(self.positions).replace("\n", ",")
        return f"{super().__repr__()[:-1]}    {positions},\n)"

    @property
    def isglobal(self):
        """Basis is defined with respect to a single (global) origin.

        Returns:
            bool
        """
        return len(self) == 0 or np.all(self.pidx == self.pidx[0])

    def __getitem__(self, idx):
        """Get a subset of the basis.

        This function allows index into the basis set by an integer, a slice, a sequence
        of integers or bools, an ellipsis, or an empty tuple. All of them except the
        integer and the empty tuple results in another basis set being returned. In case
        of the two exceptions a tuple is returned.

        Alternatively, the string "pkzmp", "kzmp", or "kzm" can be used to access only a
        subset of :attr:`pidx`, :attr:`kz`, :attr:`m`, and :attr:`pol`.
        """
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "pkzmp":
                return self.pidx, self.kz, self.m, self.pol
            elif idx == "kzmp":
                return self.kz, self.m, self.pol
            elif idx == "kzm":
                return self.kz, self.m
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.kz[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        """Compare basis sets.

        Basis sets are considered equal, when they have the same modes in the same order
        and the specified origin :attr:`positions` are equal.
        """
        try:
            return self is other or (
                np.array_equal(self.pidx, other.pidx)
                and np.array_equal(self.kz, other.kz)
                and np.array_equal(self.m, other.m)
                and np.array_equal(self.pol, other.pol)
                and np.array_equal(self.positions, other.positions)
            )
        except AttributeError:
            return False

    @classmethod
    def default(cls, kzs, mmax, nmax=1, positions=None):
        """Default basis for the given z-components of wave vector and angular momentum.

        The default order contains separate blocks for each position index which are in
        ascending order. Within each block the modes are sorted by the z-component of
        the wave vector :math:`k_z`. For each of those values the z-projection of the
        angular momentum is placed in ascending order. Finally, the polarization index
        is the fastest changing index which iterates between 1 and 0.

        Example:
            >>> CylindricalWaveBasis.default([-0.5, 0.5], 1)
            CylindricalWaveBasis(
                pidx=[0 0 0 0 0 0 0 0 0 0 0 0],
                kz=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5  0.5  0.5  0.5  0.5  0.5  0.5],
                m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[0. 0. 0.]],
            )
            >>> CylindricalWaveBasis.default([0], 1, 2, [[1., 0, 0], [-1., 0, 0]])
            CylindricalWaveBasis(
                pidx=[0 0 0 0 0 0 1 1 1 1 1 1],
                kz=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],
                m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[ 1.  0.  0.], [-1.  0.  0.]],
            )

        Args:
            kzs (array-like, float): Maximal multipolar order.
            mmax (int): Maximal value of the angular momentum z-component.
            nmax (int, optional): Number of positions, defaults to 1.
            positions (array-like, optional): Positions of the origins.
        """
        kzs = np.atleast_1d(kzs)
        if kzs.ndim > 1:
            raise ValueError(f"kzs has dimension larger than one: '{kzs.ndim}'")
        modes = [
            [n, kz, m, p]
            for n in range(nmax)
            for kz in kzs
            for m in range(-mmax, mmax + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, positions=positions)

    @classmethod
    def diffr_orders(cls, kz, mmax, lattice, bmax, nmax=1, positions=None):
        """Create a basis set for a system periodic in the z-direction.

        Example:
            >>> CylindricalWaveBasis.diffr_orders(0.1, 1, 2 * np.pi, 1)
            CylindricalWaveBasis(
                pidx=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
                kz=[-0.9 -0.9 -0.9 -0.9 -0.9 -0.9  0.1  0.1  0.1  0.1  0.1  0.1  1.1
            1.1 1.1  1.1  1.1  1.1],
                m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1 -1 -1  0  0  1  1],
                pol=[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0],
                positions=[[0. 0. 0.]],
            )

        Args:
            kz (float): Wave vector z-component. Ideally it is in the first Brillouin
                zone (use :func:`misc.firstbrillouin1d`).
            mmax (int): Maximal value for the z-component of the angular momentum.
            lattice (:class:`treams.Lattice` or float): Lattice definition or pitch.
            bmax (float): Maximal change of the z-component of the wave vector. So,
                this defines a maximal momentum transfer from the given value `kz`.
            nmax (int, optional): Number of positions.
            positions (array-like, optional): Positions of the origins.
        """
        lattice = Lattice(lattice)
        lattice_z = Lattice(lattice, "z")
        nkz = np.floor(np.abs(bmax / lattice_z.reciprocal))
        kzs = kz + np.arange(-nkz, nkz + 1) * lattice_z.reciprocal
        res = cls.default(kzs, mmax, nmax, positions=None)
        res.hints["lattice"] = lattice
        res.hints["kpar"] = [np.nan, np.nan, kz]
        return res

    def _from_iterable(cls, it, positions=None):
        hints = {}
        if isinstance(cls, CylindricalWaveBasis):
            positions = cls.positions if positions is None else positions
            hints = copy.deepcopy(cls.hints)
            cls = type(cls)
        obj = cls(it, positions=positions)
        obj.hints = hints
        return obj

    @staticmethod
    def defaultmmax(dim, nkz=1, nmax=1):
        """Calculate the default mode order for a given length.

        Given the dimension of the T-matrix return the estimated maximal value of `m`.
        This is the inverse of :meth:`defaultdim`. A value of zero is allowed for empty
        T-matrices.

        Example:
            >>> SphericalWaveBasis.defaultlmax(len(SphericalWaveBasis.default(3)))
            3

        Args:
            dim (int): Dimension of the T-matrix, respectively number of modes
            nkz (int, optional): Number of z-components of the wave vector.
            nmax (int, optional): Number of particles, defaults to 1.

        Returns:
            int
        """
        if dim % (2 * nkz * nmax) != 0:
            raise ValueError("cannot estimate the default mmax")
        dim = dim // (2 * nkz * nmax)
        if (dim - 1) % 2 != 0:
            raise ValueError("cannot estimate the default mmax")
        return (dim - 1) // 2

    @staticmethod
    def defaultdim(nkz, mmax, nmax=1):
        """Default number of modes for a given mulipolar order.

        Given the maximal value of `l` return the size of the corresponding T-matrix.
        This is the inverse of :meth:`defaultlmax`. A value of zero is allowed.

        Args:
            nkz (int): Number of z-components of the wave vector.
            mmax (int): Maximal value of the angular momentum's z-component.
            nmax (int, optional): Number of particles, defaults to 1.

        Returns:
            int
        """
        if nkz < 0 or mmax < 0:
            raise ValueError("maximal order must be positive")
        return (2 * mmax + 1) * nkz * 2 * nmax


class PlaneWaveBasis(BasisSet):
    """Plane wave basis.

    A plane wave basis is defined by a collection of wave vectors specified by the
    Cartesian wave vector components ``kx``, ``ky``, and ``kz`` and the polarizations
    ``pol``.

    For plane waves there exist multiple :ref:`polarizations:Polarizations`, such that
    these modes can refer to either :func:`~treams.special.vpw_A` or
    :func:`~treams.special.vpw_M` and :func:`~treams.special.vpw_N`.

    Args:
        modes (array-like): A tuple containing a list for each of ``kx``, ``ky``,
            ``kz``, and ``pol``.

    Attributes:
        kx (array-like): X-component of the wave vector.
        ky (array-like): Y-component of the wave vector.
        kz (array-like): Z-component of the wave vector.
        pol (array-like): Polarization, see also :ref:`polarizations:Polarizations`.
    """

    _names = ("kx", "ky", "kz", "pol")
    isglobal = True
    """A plane wave basis is always global."""

    def __init__(self, modes):
        """Initialization."""
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            kx = []
            ky = []
            kz = []
            pol = []
        elif len(modes[0]) == 4:
            kx, ky, kz, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        kx, ky, kz, pol = [np.array(i) for i in (kx, ky, kz, pol)]
        for i in (kx, ky, kz, pol):
            i.flags.writeable = False
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
        self.kx, self.ky, self.kz = kx, ky, kz
        self.pol = np.array(pol.real, int)
        if np.any(self.pol != pol):
            raise ValueError("polarizations must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        self.hints = {}

    def __getitem__(self, idx):
        """Get a subset of the basis.

        This function allows index into the basis set by an integer, a slice, a sequence
        of integers or bools, an ellipsis, or an empty tuple. All of them except the
        integer and the empty tuple results in another basis set being returned. In case
        of the two exceptions a tuple is returned.

        Alternatively, the string "xyzp", "xyp", or "zp" can be used to access only a
        subset of :attr:`kx`, :attr:`ky`, :attr:`kz`, and :attr:`pol`.
        """
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "xyzp":
                return self.kx, self.ky, self.kz, self.pol
            elif idx == "xyp":
                return self.kx, self.ky, self.pol
            elif idx == "zp":
                return self.kz, self.pol
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.kx[idx], self.ky[idx], self.kz[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res))

    @classmethod
    def default(cls, kvecs):
        """Default basis from the given wave vectors.

        For each wave vector the two polarizations 1 and 0 are taken.

        Example:
            >>> PlaneWaveBasis.default([[0, 0, 5], [0, 3, 4]])
            PlaneWaveBasis(
                kx=[0 0 0 0],
                ky=[0 0 3 3],
                kz=[5 5 4 4],
                pol=[1 0 1 0],
            )

        Args:
            kvecs (array-like): Wave vectors in Cartesian coordinates.
        """
        kvecs = np.atleast_2d(kvecs)
        modes = np.empty((2 * kvecs.shape[0], 4), kvecs.dtype)
        modes[::2, :3] = kvecs
        modes[1::2, :3] = kvecs
        modes[::2, 3] = 1
        modes[1::2, 3] = 0
        return cls(modes)

    def _from_iterable(cls, it):
        hints = {}
        if isinstance(cls, PlaneWaveBasis):
            hints = copy.deepcopy(cls.hints)
            cls = type(cls)
        obj = cls(it)
        obj.hints = hints
        return obj

    def __eq__(self, other):
        """Compare basis sets.

        Basis sets are considered equal, when they have the same modes in the same
        order.
        """
        try:
            return self is other or (
                np.array_equal(self.kx, other.kx)
                and np.array_equal(self.ky, other.ky)
                and np.array_equal(self.kz, other.kz)
                and np.array_equal(self.pol, other.pol)
            )
        except AttributeError:
            return False

    def partial(self, alignment="xy", k0=None, material=Material()):
        """Create a :class:`PlaneWavePartial`.

        The plane wave basis is changed to a partial basis, where only two (real-valued)
        wave vector components are defined the third component is then inferred from the
        dispersion relation, which depends on the wave number and material, and a
        ``modetype`` that specifies the sign of the third component.

        Args:
            alignment (str, optional): Wave vector components that are part of the
                partial basis. Defaults to "xy", other permitted values are "yz" and
                "zx".
            k0 (float, optional): Wave number. If given, it is checked that the current
                basis fulfils the dispersion relation.
            material (:class:`~treams.Material` or tuple): Material definition. Defaults
                to vacuum/air.
        """
        if k0 is not None:
            ks = material.ks(k0)[self.pol]
            test = self.kx * self.kx + self.ky * self.ky + self.kz * self.kz - ks * ks
            if np.any(
                np.logical_or(np.abs(test.real) > 1e-10, np.abs(test.imag) > 1e-10)
            ):
                raise ValueError("dispersion relation violated")
        alignment = "xy" if alignment is None else alignment
        if alignment in ("xy", "yz", "zx"):
            kpar = [getattr(self, "k" + s) for s in alignment]
        else:
            raise ValueError(f"invalid alignment '{alignment}'")
        obj = PlaneWaveBasisPartial(zip(*kpar, self.pol), alignment)
        obj.hints = copy.deepcopy(self.hints)
        return obj

    def complete(self, k0, material=Material(), modetype=None):
        """Create a complete basis.

        A plane wave basis is considered complete, when all three Cartesian components
        and the polarization is defined for each mode. The plane wave basis of this
        class is already complete, so it. Child classes might override this function to
        create a complete :class:`PlaneWaveBasis`. Here, it only performs checks on the
        dispersion relation.

        Args:
            k0 (float): Wave number.
            material (:class:`~treams.Material` or tuple, optional): Material
                definition. Defaults to vacuum/air.
            modetype (optional): Currently unused for this class.
        """
        # TODO: check kz depending on modetype (alignment?)
        ks = (k0 * Material(material).nmp)[self.pol]
        if (np.abs(self.kx ** 2 + self.ky ** 2 + self.kz ** 2 - ks * ks) > 1e-14).any():
            raise ValueError("incompatible k0 and/or material")
        return self


class PlaneWaveBasisPartial(PlaneWaveBasis):
    """Partial plane wave basis.

    A partial plane wave basis is defined by two wave vector components out of all three
    Cartesian wave vector components ``kx``, ``ky``, and ``kz`` and the polarizations
    ``pol``. Which two components are given is specified in the :attr:`alignment`. This
    basis is mostly used for stratified media that are periodic or uniform in the two
    alignment directions, such that the given wave vector components correspond to the
    diffraction orders.

    For plane waves there exist multiple :ref:`polarizations:Polarizations`, such that
    these modes can refer to either :func:`~treams.special.vpw_A` or
    :func:`~treams.special.vpw_M` and :func:`~treams.special.vpw_N`.

    Args:
        modes (array-like): A tuple containing a list for each of ``k1``, ``k2``, and
            ``pol``.
        alignment (str, optional): Definition which wave vector components are given.
            Defaults to "xy", other possible values are "yz" and "zx".

    Attributes:
        alignment (str): Alignment of the partial basis.
        pol (array-like): Polarization, see also :ref:`polarizations:Polarizations`.
    """

    def __init__(self, modes, alignment="xy"):
        """Initialization."""
        alignment = "xy"
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        tmp = []
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            kx = []
            ky = []
            pol = []
        elif len(modes[0]) == 3:
            kx, ky, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        self._kx, self._ky = [np.real(i) for i in (kx, ky)]
        self.pol = np.array(np.real(pol), int)
        for i, j in [(self._kx, kx), (self._ky, ky), (self.pol, pol)]:
            i.flags.writeable = False
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
            if np.any(i != j):
                raise ValueError("invalid value for parameter, must be real")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")

        if alignment in ("xy", "yz", "zx"):
            self._names = (*("k" + i for i in alignment),) + ("pol",)
        else:
            raise ValueError(f"invalid alignment '{alignment}'")

        self.alignment = alignment
        self.hints = {}

    @property
    def kx(self):
        """X-components of the wave vector.

        If the components are not specified `None` is returned.
        """
        if self.alignment == "xy":
            return self._kx
        elif self.alignment == "zx":
            return self._ky
        return None

    @property
    def ky(self):
        """Y-components of the wave vector.

        If the components are not specified `None` is returned.
        """
        if self.alignment == "xy":
            return self._ky
        elif self.alignment == "yz":
            return self._kx
        return None

    @property
    def kz(self):
        """Z-components of the wave vector.

        If the components are not specified `None` is returned.
        """
        if self.alignment == "yz":
            return self._ky
        elif self.alignment == "zx":
            return self._kx
        return None

    def __getitem__(self, idx):
        """Get a subset of the basis.

        This function allows index into the basis set by an integer, a slice, a sequence
        of integers or bools, an ellipsis, or an empty tuple. All of them except the
        integer and the empty tuple results in another basis set being returned. In case
        of the two exceptions a tuple is returned.
        """
        res = self._kx[idx], self._ky[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res))

    @classmethod
    def default(cls, kpars, alignment="xy"):
        """Default basis from the given wave vectors.

        For each wave vector the two polarizations 1 and 0 are taken.

        Example:
            >>> PlaneWaveBasisPartial.default([[0, 0], [0, 3]])
            PlaneWaveBasisPartial(
                kx=[0. 0. 0. 0.],
                ky=[0. 0. 3. 3.],
                pol=[1 0 1 0],
            )

        Args:
            kpars (array-like): Wave vector components in Cartesian coordinates.
            alignment (str, optional): Definition which wave vector components are
                given. Defaults to "xy", other possible values are "yz" and "zx".

        """
        kpars = np.atleast_2d(kpars)
        modes = np.empty((2 * kpars.shape[0], 3), float)
        modes[::2, :2] = kpars
        modes[1::2, :2] = kpars
        modes[::2, 2] = 1
        modes[1::2, 2] = 0
        return cls(modes, alignment=alignment)

    @classmethod
    def diffr_orders(cls, kpar, lattice, bmax):
        """Create a basis set for a two-dimensional periodic system.

        The reciprocal lattice to the given lattice is taken to consider all diffraction
        orders that lie within the defined maximal radius (in reciprocal space).

        Example:
            >>> PlaneWaveBasisPartial.diffr_orders([0, 0], Lattice.square(2 * np.pi), 1)
            PlaneWaveBasisPartial(
                kx=[ 0.  0.  0.  0.  0.  0.  1.  1. -1. -1.],
                ky=[ 0.  0.  1.  1. -1. -1.  0.  0.  0.  0.],
                pol=[1 0 1 0 1 0 1 0 1 0],
            )

        Args:
            kpar (float): Tangential wave vector components. Ideally they are in the
                first Brillouin zone (use :func:`misc.firstbrillouin2d`).
            lattice (:class:`treams.Lattice` or float): Lattice definition or pitch.
            bmax (float): Maximal change of tangential wave vector components. So,
                this defines a maximal momentum transfer.
        """
        lattice = Lattice(lattice)
        if lattice.dim != 2:
            raise ValueError("invalid lattice dimensions")
        latrec = lattice.reciprocal
        kpars = kpar + la.diffr_orders_circle(latrec, bmax) @ latrec
        obj = cls.default(kpars, alignment=lattice.alignment)
        obj.hints["lattice"] = lattice
        if lattice.alignment == "xy":
            kpar = list(kpar) + [np.nan]
        elif lattice.alignment == "yz":
            kpar = [np.nan] + list(kpar)
        else:
            kpar = [kpar[1], np.nan, kpar[0]]
        obj.hints["kpar"] = kpar
        return obj

    def _from_iterable(cls, it, alignment="xy"):
        if isinstance(cls, PlaneWaveBasisPartial):
            alignment = cls.alignment if alignment is None else alignment
        obj = super()._from_iterable(it)
        obj.alignment = obj.alignment if alignment is None else alignment
        return obj

    def __eq__(self, other):
        """Compare basis sets.

        Basis sets are considered equal, when they have the same modes in the same
        order.
        """
        try:
            skx, sky, skz = self.kx, self.ky, self.kz
            okx, oky, okz = other.kx, other.ky, other.kz
            return self is other or (
                (np.array_equal(skx, okx) or (skx is None and okx is None))
                and (np.array_equal(sky, oky) or (sky is None and oky is None))
                and (np.array_equal(skz, okz) or (skz is None and okz is None))
                and np.array_equal(self.pol, other.pol)
            )
        except AttributeError:
            return False

    def complete(self, k0, material=Material(), modetype="up"):
        """Create a complete basis :class:`PlaneWaveBasis`.

        A plane wave basis is considered complete, when all three Cartesian components
        and the polarization is defined for each mode. So, the specified wave number,
        material, and modetype is taken to calculate the third Cartesian wave vector.
        The modetype "up" ("down") is for waves propagating in the positive (negative)
        direction with respect to the Cartesian axis that is orthogonal to the
        alignment.

        Args:
            k0 (float): Wave number.
            material (:class:`~treams.Material` or tuple, optional): Material
                definition. Defaults to vacuum/air.
            modetype (str, optional): Propagation direction. Defaults to "up".
        """
        if modetype not in ("up", "down"):
            raise ValueError("modetype not recognized")
        material = Material(material)
        kx = self._kx
        ky = self._ky
        kz = material.kzs(k0, kx, ky, self.pol) * (2 * (modetype == "up") - 1)
        if self.alignment == "yz":
            kx, ky, kz = kz, kx, ky
        elif self.alignment == "zy":
            kx, ky, kz = ky, kz, kx
        obj = PlaneWaveBasis(zip(kx, ky, kz, self.pol))
        obj.hints = copy.deepcopy(self.hints)
        return obj


def _raise_basis_error(*args):
    raise TypeError("'basis' must be BasisSet")


class PhysicsDict(util.AnnotationDict):
    """Physics dictionary.

    Derives from :class:`treams.util.AnnotationDict`. This dictionary has additionally
    several properties defined.

    Attributes:
        basis (:class:`BasisSet`): Basis set.
        k0 (float): Wave number.
        kpar (list): Parallel wave vector components. Usually, this is a list of length
            3 with its items corresponding to the Cartesian axes. Unspecified items are
            set to `nan`.
        lattice (:class:`~treams.Lattice`): Lattice definition.
        material (:class:`~treams.Material`): Material definition.
        modetype (str): Mode type, for spherical and cylindrical waves this can be
            "incident" and "scattered", for partial plane waves it can be "up" or
            "down".
        poltype (str): Polarization, see also :ref:`polarizations:Polarizations`.
    """

    properties = {
        "k0": ("Wave number.", lambda x: isinstance(x, float), float),
        "basis": (
            ":class:`~treams.BasisSet`.",
            lambda x: isinstance(x, BasisSet),
            _raise_basis_error,
        ),
        "poltype": (
            ":ref:`polarizations:Polarizations`.",
            lambda x: isinstance(x, str),
            str,
        ),
        "modetype": ("Mode type.", lambda x: isinstance(x, str), str),
        "material": (
            ":class:`~treams.Material`.",
            lambda x: isinstance(x, Material),
            Material,
        ),
        "lattice": (
            ":class:`~treams.Lattice`.",
            lambda x: isinstance(x, Lattice),
            Lattice,
        ),
        "kpar": (
            "Wave vector components tangential to the lattice.",
            lambda x: isinstance(x, list),
            list,
        ),
    }
    """Special properties tracked by the PhysicsDict."""

    def __setitem__(self, key, val):
        """Set item specified by key to the defined value.

        When overwriting an existing key an :class:`AnnotationWarning` is emitted.
        Avoid the warning by explicitly deleting the key first. The special attributes
        are cast to their corresponding types.

        Args:
            key (hashable): Key
            val : Value

        Warns:
            AnnotationWarning
        """
        _, testfunc, castfunc = self.properties.get(key, (lambda x: True, None))
        if not testfunc(val):
            val = castfunc(val)
        super().__setitem__(key, val)


@util.register_properties
class PhysicsArray(util.AnnotatedArray):
    """Physics-aware array.

    A physics aware array is a special type of :class`~treams.util.AnnotatedArray`.
    Additionally to keeping track of the annotations, it is enhanced by the ability to
    create suiting linear operators to perform tasks like rotations, translations, or
    expansions into different basis sets and by applying special rules for the
    physical properties of :class:`PhysicsDict` upon matrix multiplications, see also
    :meth:`__matmul__`.
    """

    _scales = {"basis"}
    _properties = PhysicsDict.properties

    changepoltype = op.ChangePoltype()
    """Polarization change matrix, see also :class:`treams.operators.ChangePoltype`."""
    efield = op.EField()
    """Field evaluation matrix, see also :class:`treams.operators.EField`."""
    expand = op.Expand()
    """Expansion matrix, see also :class:`treams.operators.Expand`."""
    expandlattice = op.ExpandLattice()
    """Lattice expansion matrix, see also :class:`treams.operators.ExpandLattice`."""
    permute = op.Permute()
    """Permutation matrix, see also :class:`treams.operators.Permute`."""
    rotate = op.Rotate()
    """Rotation matrix, see also :class:`treams.operators.Rotate`."""
    translate = op.Translate()
    """Translation matrix, see also :class:`treams.operators.Translate`."""

    def __init__(self, arr, ann=(), **kwargs):
        """Initialization."""
        super().__init__(arr, ann, **kwargs)
        self._check()

    @property
    def ann(self):
        """Array annotations."""
        # necessary to define the setter below
        return super().ann

    @ann.setter
    def ann(self, ann):
        """Set array annotations.

        See also :meth:`treams.util.AnnotatedArray.__setitem__`.
        """
        self._ann = util.AnnotationSequence(*(({},) * self.ndim), mapping=PhysicsDict)
        self._ann.update(ann)

    def __repr__(self):
        """String representation.

        For a more managable output only the special physics properties are shown
        alongside the array itself.
        """
        repr_arr = "    " + repr(self._array)[6:-1].replace("\n  ", "\n")
        repr_kwargs = ",\n    ".join(
            f"{key}={repr(getattr(self, key))}"
            for key in self._properties
            if getattr(self, key) is not None
        )
        if repr_kwargs != "":
            repr_arr += ",\n    " + repr_kwargs
        return f"{self.__class__.__name__}(\n{repr_arr}\n)"

    def _check(self):
        """Run checks to validate the physical properties.

        The checks are run on the last two dimensions. They include:

            * Dispersion relation checks for :class:`PlaneWaveBasis` if `basis`, `k0`
              and `material` is defined`
            * `parity` polarizations are not physically permitted in chiral media.
            * All lattices explicitly given and in the basis hints must be compatible.
            * All tangential wave vector compontents must be compatible.
        """
        total_lat = None
        total_kpar = [np.nan] * 3
        for a in self.ann[-2:]:
            k0 = a.get("k0")
            material = a.get("material")
            modetype = a.get("modetype")
            poltype = a.get("poltype")
            basis = a.get("basis")
            lattice = a.get("lattice")
            for lat in (
                lattice,
                getattr(a.get("basis"), "hints", {}).get("lattice"),
            ):
                if lat is not None:
                    total_lat = lat | total_lat
            for kpar in (
                a.get("kpar"),
                getattr(a.get("basis"), "hints", {}).get("kpar"),
            ):
                if kpar is not None:
                    for i, (x, y) in enumerate(zip(total_kpar, kpar)):
                        if np.isnan(x):
                            total_kpar[i] = y
                        elif not np.isnan(y) and x != y:
                            raise ValueError("incompatible kpar")
            if type(basis) == PlaneWaveBasis and None not in (k0, material):
                basis.complete(k0, material, modetype)
            if poltype == "parity" and getattr(material, "ischiral", False):
                raise ValueError("poltype 'parity' not permitted for chiral material")

    def __matmul__(self, other, *args, **kwargs):
        """Matrix multiplication.

        When one of the operands for the matrix multiplication does not contain a
        physical property that is present in the other matrix, it becomes "transparent"
        to it, meaning that it is just copied to the corresponding results direction.

        Example:
            >>> res = PhysicsArray([0, 1], k0=1.5, foo="bar") @ [[2, 3], [4, 5]]
            >>> res.ann
            AnnotationSequence(PhysicsDict({'k0': 1.5}),)

            Note that the result contains `k0` but not `foo`, since the "transparency"
            of the second array only applies to physical properties.
        """
        res = super().__matmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", ({}, {}))
        other_ndim = np.ndim(other)
        for name in self._properties:
            if self.ndim > 1 and self.ann[-1].get(name) is None:
                dim = -1 - (other_ndim != 1)
                val = other_ann[dim].get(name)
                if val is not None:
                    res.ann[dim][name] = val
            val = self.ann[-1].get(name)
            if other_ndim > 1 and other_ann[-2].get(name) is None and val is not None:
                res.ann[-1][name] = val
        return res

    def __rmatmul__(self, other, *args, **kwargs):
        """Matrix multiplication with opposite operand order.

        See also, :meth:`__matmul__`.
        """
        res = super().__rmatmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", ({},))
        other_ndim = np.ndim(other)
        for name in self._properties:
            if other_ndim > 1 and other_ann[-1].get(name) is None:
                dim = -1 - (self.ndim != 1)
                val = self.ann[dim].get(name)
                if val is not None:
                    res.ann[dim][name] = val
            val = other_ann[-1].get(name)
            if self.ndim > 1 and self.ann[-2].get(name) is None and val is not None:
                res.ann[-1][name] = val
        return res
