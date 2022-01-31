import warnings

import numpy as np

from ptsa import misc


class TMatrixBase:
    """
    T-matrix base class

    The T-matrix is square, with the modes defined in the corresponding fields. The
    order of the T-matrix can be arbitrary. Helicity and parity modes are possible, but
    not mixed.

    The embedding medium is described by permittivity, permeability and the chirality
    parameter.

    The T-matrix can be global or local. For a local T-matrix multiple positions have to
    be specified. Also modes must have as first element a position index.

    Note:

        This class is not intended for any direct use, its purpose is to collect common
        features of all T-matrices.

    Args:
        tmat (float or complex, array): T-matrix itself
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        positions (float, (3,)- or (M,3)-array): Positions for a local T-matrix
        helicity (bool): Helicity or parity modes
        modes (iterable): Sorting of the T-matrix entries. Either four entries for
            local T-Matrices, with the first specifying the corresponding position
            or three entries only, specifying degree, order, and polarization.

    Attributes:
        t (float or complex, (N,N)-array): T-matrix
        k0 (float): Wave number in vacuum
        positions (float, (3,)- or (M,3)-array): Positions for a local T-matrix
        epsilon (float or complex, optional): Relative permittivity of the embedding medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        helicity (bool): Helicity or parity modes
        ks (float or complex (2,)-array): Wave numbers in the medium for both polarizations
    """

    def __init__(
        self,
        tmat,
        k0,
        epsilon=1,
        mu=1,
        kappa=0,
        positions=None,
        helicity=True,
        modes=None,
    ):
        if kappa != 0 and not helicity:
            raise ValueError("chiral medium requires helicity modes")
        if positions is None:
            positions = np.zeros((1, 3))
        positions = np.array(positions)
        tmat = np.array(tmat)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError(
                f"invalid T-matrix shape {tmat.shape}, expected square matrix"
            )
        self.t = tmat
        self.k0 = np.array(k0, float).item()
        self.epsilon = np.array(epsilon).item()
        self.mu = np.array(mu).item()
        self.kappa = np.array(kappa).item()
        self.positions = positions
        self.helicity = helicity
        self.ks = k0 * misc.refractive_index(epsilon, mu, kappa)

    @classmethod
    def cluster(cls, tmats, positions):
        """
        Block-diagonal collection of T-matrices

        This is the starting point for a local T-matrix. To finally get a the local
        T-Matrix, calculate the interaction.

        Args:
            tmats: T-matrices
            positions: The position of each T-matrix

        Returns:
            TMatrixBase
        """
        positions = np.array(positions)
        if len(tmats) < positions.shape[0]:
            warnings.warn("specified more positions than T-matrices")
        elif len(tmats) > positions.shape[0]:
            raise ValueError(
                f"got {len(tmats)} T-matrices and only {positions.shape[0]} positions"
            )
        material = tmats[0].material
        k0 = tmats[0].k0
        helicity = tmats[0].helicity
        dim = sum([tmat.t.shape[0] for tmat in tmats])
        modes = [np.zeros((dim,), int) for _ in range(4)]
        tlocal = np.zeros((dim, dim), complex)
        i = 0
        for j, tmat in enumerate(tmats):
            if tmat.material != material:
                warnings.warn(f"materials {material} and {tmat.material} do not match")
            if tmat.k0 != k0:
                warnings.warn(f"vacuum wave numbers {k0} and {tmat.k0} do not match")
            if tmat.helicity != helicity:
                raise ValueError("found T-matrices in helicity and parity mode")
            dim = tmat.t.shape[0]
            for k in range(1, 4):
                modes[k][i : i + dim] = tmat.modes[k - 1]
            modes[0][i : i + dim] = j
            tlocal[i : i + dim, i : i + dim] = tmat.t
            i += dim
        return cls(tlocal, k0, *material, positions, helicity, modes)

    @property
    def material(self):
        """
        Material parameters of the embedding medium

        The relative permittivity, relative permeability, and the chirality parameter
        specify the embedding medium.

        Returns:
            tuple
        """
        return self.epsilon, self.mu, self.kappa

    @property
    def fullmodes(self):
        """
        Positions and modes of the T-matrix

        Position and mode indices of each row/column in the T-matrix

        Returns:
            4-tuple
        """
        return (self.pidx, *self.modes)

    def changebasis(self, modes=None):
        """
        Swap between helicity and parity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            TMatrixBase
        """
        if self.helicity and self.kappa != 0:
            raise ValueError(
                "Basis change not possible for non-zero kappa"
            )  # todo: Fix error type
        if modes is None:
            modes = self.fullmodes
        modes = self._check_modes(modes)
        mat = misc.basischange(self.fullmodes, modes)
        self.t = mat.T @ self.t @ mat
        self.pidx, self.l, self.m, self.pol = modes
        self.helicity = not self.helicity
        return self

    def helicitybasis(self, modes=None):
        """
        Change to helicity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            TMatrixBase
        """
        if not self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def paritybasis(self, modes=None):
        """
        Change to parity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            TMatrixBase
        """
        if self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def interact(self):
        """
        Interact the blockdiagonal T-matrices to a local T-matrix

        This function calculates the coupling between the T-matrices and then solves
        the interaction between the different T-matrices. This function changes the
        T-matrix.

        Returns:
            TMatrixBase
        """
        self.t = np.linalg.solve(self.coupling(), self.t)
        return self

    def latticeinteract(self, kpar, a, eta=0):
        r"""
        The T-matrix after interaction with the lattice.

        Returns

        .. math::

            (\mathbb 1 - T C)^{-1} T

        Args:
            kpar (float): The parallel component of the T-matrix
            a (array): Definition of the lattice
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            TMatrixBase
        """
        self.t = np.linalg.solve(self.latticecoupling(kpar, a, eta), self.t)

    def _check_modes(self, modes):
        """_check_modes"""
        if len(modes) < 3 or len(modes) > 4:
            raise ValueError(f"invalid length of variable modes {len(modes)}, must be 3 or 4")
        modes = (*(np.array(a) for a in modes),)
        if len(modes) == 3:
            modes = (np.zeros_like(modes[0]),) + modes
        if not np.all([m.ndim == 1 for m in modes]):
            raise ValueError("invalid dimensions of modes")
        if not np.all([m.size == modes[0].size for m in modes[1:]]):
            raise ValueError("all modes need equal size")
        return modes
