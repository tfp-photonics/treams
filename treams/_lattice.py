import collections

import numpy as np

import treams.lattice as la


class Lattice:
    """Lattice definition.

    The lattice can be one-, two-, or three-dimensional. If it is not three-dimensional
    it is required to be embedded into a lower dimensional subspace that is aligned with
    the Cartesian axes. The default alignment for one-dimensional lattices is along the
    z-axis and for two-dimensional lattices it is in the x-y-plane.

    For a one-dimensional lattice the definition consists of simply the value of the
    lattice pitch. Higher-dimensional lattices are defined by (2, 2)- or (3, 3)-arrays.
    If the arrays are diagonal it is sufficient to specify the (2,)- or (3,)-array
    diagonals. Alternatively, if another instance of a Lattice is given, the defined
    alignment is extracted, which can be used to separate a lower-dimensional
    sublattice.

    Lattices are immutable objects.

    Example:
        >>> Lattice([1, 2])
        Lattice([[1. 0.]
                 [0. 2.]], alignment='xy')
        >>> Lattice(_, 'x')
        Lattice(1.0, alignment='x')

    Args:
        arr (float, array, Lattice): Lattice definition. Each row corresponds to one
            lattice vector, each column to the axis defined in `alignment`.
        alignment (str, optional): Alignment of the lattice. Possible values are 'x',
            'y', 'z', 'xy', 'yz', 'zx', and 'xyz'.
    """

    _allowed_alignments = {
        "x": "y",
        "y": "z",
        "z": "x",
        "xy": "yz",
        "yz": "zx",
        "zx": "xy",
        "xyz": "xyz",
    }

    def __init__(self, arr, alignment=None):
        """Initialization."""
        if isinstance(arr, Lattice):
            self._alignment = arr.alignment if alignment is None else alignment
            self._lattice = arr._sublattice(self._alignment)[...]
            self._reciprocal = (
                2 * np.pi / self[...] if self.dim == 1 else la.reciprocal(self[...])
            )
            return
        if arr is None:
            raise ValueError("Lattice cannot be 'None'")

        arr = np.array(arr, float)
        arr.flags.writeable = False
        alignments = {1: ("z", "x", "y"), 4: ("xy", "yz", "zx"), 9: ("xyz",)}
        if arr.ndim < 3 and arr.size == 1:
            self._lattice = np.squeeze(arr)
        elif arr.ndim == 1 and arr.size in (2, 3):
            self._lattice = np.diag(arr)
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] in (2, 3):
            self._lattice = arr
        else:
            raise ValueError(f"invalid shape '{arr.shape}'")

        alignment = (
            alignments[self._lattice.size][0] if alignment is None else alignment
        )
        if alignment not in alignments[self._lattice.size]:
            raise ValueError(f"invalid alignment '{alignment}'")
        self._alignment = alignment
        if self.volume == 0:
            raise ValueError("linearly dependent lattice vectors")
        self._reciprocal = (
            2 * np.pi / self[...] if self.dim == 1 else la.reciprocal(self[...])
        )

    @property
    def alignment(self):
        """The alignment of the lattice in three-dimensional space.

        For three-dimensional lattices it is always 'xyz' but lower-dimensional lattices
        have to be aligned with a subset of the axes.

        Returns:
            str
        """
        return self._alignment

    @classmethod
    def square(cls, pitch, alignment=None):
        """Create a two-dimensional square lattice.

        Args:
            pitch (float): Lattice constant.
            alignment (str, optional): Alignment of the two-dimensional lattice in the
                three-dimensional space. Defaults to 'xy'.

        Returns:
            Lattice
        """
        return cls(2 * [pitch], alignment)

    @classmethod
    def cubic(cls, pitch, alignment=None):
        """Create a three-dimensional cubic lattice.

        Args:
            pitch (float): Lattice constant.
            alignment (str, optional): Alignment of the lattice. Defaults to 'xyz'.

        Returns:
            Lattice
        """
        return cls(3 * [pitch], alignment)

    @classmethod
    def rectangular(cls, x, y, alignment=None):
        """Create a two-dimensional rectangular lattice.

        Args:
            x (float): Lattice constant along the first dimension. For the default
                alignment this corresponds to the x-axis.
            y (float): Lattice constant along the second dimension. For the default
                alignment this corresponds to the y-axis.
            alignment (str, optional): Alignment of the two-dimensional lattice in the
                three-dimensional space. Defaults to 'xy'.

        Returns:
            Lattice
        """
        return cls([x, y], alignment)

    @classmethod
    def orthorhombic(cls, x, y, z, alignment=None):
        """Create a three-dimensional orthorhombic lattice.

        Args:
            x (float): Lattice constant along the first dimension. For the default
                alignment this corresponds to the x-axis.
            y (float): Lattice constant along the second dimension. For the default
                alignment this corresponds to the y-axis.
            z (float): Lattice constant along the third dimension. For the default
                alignment this corresponds to the z-axis.
            alignment (str, optional): Alignment of the lattice. Defaults to 'xyz'.

        Returns:
            Lattice
        """
        return cls([x, y, z], alignment)

    @classmethod
    def hexagonal(cls, pitch, height=None, alignment=None):
        """Create a hexagonal lattice.

        The lattice is two-dimensional if no height is specified else it is
        three-dimensional

        Args:
            pitch (float): Lattice constant.
            height (float, optional): Separation along the third axis for a
                three-dimensional lattice.
            alignment (str, optional): Alignment of the two-dimensional lattice in the
                three-dimensional space. Defaults to either 'xy' or 'xyz' in the two-
                or three-dimensional case, respectively.

        Returns:
            Lattice
        """
        if height is None:
            return cls(
                np.array([[pitch, 0], [0.5 * pitch, np.sqrt(0.75) * pitch]]), alignment
            )
        return cls(
            np.array(
                [[pitch, 0, 0], [0.5 * pitch, np.sqrt(0.75) * pitch, 0], [0, 0, height]]
            ),
            alignment,
        )

    def __eq__(self, other):
        """Equality.

        Two lattices are considered equal when they have the same dimension, alignment,
        and lattice vectors.

        Args:
            other (Lattice): Lattice to compare with.

        Returns:
            bool
        """
        return other is not None and (
            self is other
            or (
                self.alignment == other.alignment
                and self.dim == other.dim
                and np.all(self[...] == other[...])
            )
        )

    @property
    def dim(self):
        """Dimension of the lattice.

        Returns:
            int
        """
        if self._lattice.ndim == 0:
            return 1
        return self._lattice.shape[0]

    @property
    def volume(self):
        """(Generalized) volume of the lattice.

        The value gives the lattice pitch, area, or volume depending on its dimension.
        The volume is signed.

        Returns:
            float
        """
        if self.dim == 1:
            return self._lattice
        return la.volume(self._lattice)

    @property
    def reciprocal(self):
        r"""Reciprocal lattice.

        The reciprocal lattice to a given lattice with dimension :math:`d` and lattice
        vectors :math:`\boldsymbol a_i` for :math:`i \in \{1, \dots, d\}` is defined by
        lattice vectors :math:`\boldsymbol b_j` with :math:`j \in \{1, \dots, d\}` such
        that :math:`\boldsymbol a_i \boldsymbol b_j = 2 \pi \delta_{ij}` is fulfilled.

        Returns:
            Lattice
        """
        return self._reciprocal

    def __getitem__(self, idx):
        """Index into the lattice.

        Indexing can be used to obtain entries from the lattice vector definitions or
        by using the Ellipsis or empty tuple to obtain the full array.

        Returns:
            float
        """
        return self._lattice[idx]

    def __str__(self):
        """String representation.

        Simply returns the lattice pitch or lattice vectors.

        Returns:
            str
        """
        return str(self._lattice)

    def __repr__(self):
        """Representation.

        The result can be used to recreate an equal instance.

        Returns:
            str
        """
        string = str(self._lattice).replace("\n", "\n        ")
        return f"Lattice({string}, alignment='{self.alignment}')"

    def _sublattice(self, key):
        """Get the sublattice defined by key.

        This function is called when one gives another instance of Lattice to the
        constructor.

        Args:
            key (str): Alignment of the sublattice to extract.

        Returns:
            Lattice
        """
        key = key.lower()
        if self.dim == 1:
            if key == self.alignment:
                return Lattice(self[...], key)
            raise ValueError(f"sublattice with key '{key}' not availale")
        idx = []
        for c in key:
            idx.append(self.alignment.find(c))
        if -1 in idx or key not in self._allowed_alignments:
            raise ValueError(f"sublattice with key '{key}' not availale")
        idx_opp = [i for i in range(self.dim) if i not in idx]
        mask = np.any(self[:, idx] != 0, axis=-1)
        if (
            (self[mask, :][:, idx_opp] != 0).any()
            or (self[np.logical_not(mask), :][:, idx] != 0).any()
            or len(idx) != sum(mask)
        ):
            raise ValueError("cannot determine sublattice")
        return Lattice(self[mask, :][:, idx], key)

    def permute(self, n=1):
        """Permute the lattice orientation.

        Get a new lattice with the alignment permuted.

        Examples:
            >>> Lattice.hexagonal(1, 2).permute()
            Lattice([[0.        1.        0.       ]
                     [0.        0.5       0.8660254]
                     [2.        0.        0.       ]], alignment='xyz')
            >>> Lattice.hexagonal(1).permute()
            Lattice([[1.        0.       ]
                     [0.5       0.8660254]], alignment='yz')

        Args:
            n (int, optional): Number of repeated permutations. Defaults to `1`.

        Returns:
            Lattice
        """
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        lattice = self[...]
        alignment = self.alignment
        if self.dim == 3:
            while n > 0:
                lattice = lattice[:, [2, 0, 1]]
                n -= 1
        else:
            while n > 0:
                alignment = self._allowed_alignments[alignment]
                n -= 1
        return Lattice(lattice, alignment)

    def __bool__(self):
        """Lattice instances always equate to True.

        Returns:
            bool
        """
        return True

    def __or__(self, other):
        """Merge two lattices.

        Two lattices are combined into one if possible.

        Example:
            >>> Lattice(1, 'x') | Lattice(2)
            Lattice([[2. 0.]
                     [0. 1.]], alignment='zx')

        Args:
            other (Lattice): Lattice to merge.

        Returns:
            Lattice
        """
        if other is None or self == other:
            return Lattice(self)

        alignment = list({c for lat in (self, other) for c in lat.alignment})
        alignment = "".join(sorted(alignment))
        if alignment == "xz":
            alignment = "zx"

        ndim = len(alignment)

        if ndim == 2:
            if self.dim == 1 == other.dim:
                if alignment.find(self.alignment) == 0:
                    return Lattice([self[...], other[...]], alignment)
                return Lattice([other[...], self[...]], alignment)
            if self.dim == 2 and Lattice(self, other.alignment) == other:
                return Lattice(self)
            if other.dim == 2 and Lattice(other, self.alignment) == self:
                return Lattice(other)
        elif ndim == 3:
            if self.dim == 3 and Lattice(self, other.alignment) == other:
                return Lattice(self)
            if other.dim == 3 and Lattice(other, self.alignment) == self:
                return Lattice(other)
            if self.dim == 2 == other.dim:
                arr = np.zeros(3)
                for i, c in enumerate("xyz"):
                    if c in self.alignment:
                        la0 = Lattice(self, c)
                    else:
                        la0 = None
                    if c in other.alignment:
                        la1 = Lattice(other, c)
                    else:
                        la1 = None
                    if None not in (la0, la1) and la0 != la1:
                        raise ValueError("cannot combine lattices")
                    arr[i] = la0[...] if la0 is not None else la1[...]
                return Lattice(arr)
            arr = np.zeros((3, 3))
            la0, la1 = (self, other) if self.dim == 1 else (other, self)
            if la0.alignment == "x":
                arr[0, 0] = la0[...]
                arr[1:, 1:] = la1[...]
                return Lattice(arr)
            if la0.alignment == "y":
                arr[1, 1] = la0[...]
                arr[[[2], [0]], [2, 0]] = la1[...]
                return Lattice(arr)
            if la0.alignment == "z":
                arr[2, 2] = la0[...]
                arr[:2, :2] = la1[...]
                return Lattice(arr)
        raise ValueError("cannot combine lattices")

    def __and__(self, other):
        """Intersect two lattices.

        The intersection of two lattices is taken if possible.

        Example:
            >>> Lattice([1, 2]) & Lattice([2, 3], 'yz')
            Lattice(2.0, alignment='y')

        Args:
            other (Lattice): Lattice to intersect.

        Returns:
            Lattice
        """
        if other is None:
            return None
        if self == other:
            return Lattice(self)

        alignment = list({c for c in self.alignment if c in other.alignment})
        if len(alignment) == 0:
            raise ValueError("cannot combine lattices")
        alignment = "".join(sorted(alignment))
        if alignment == "xz":
            alignment = "zx"

        a, b = Lattice(self, alignment), Lattice(other, alignment)
        if a == b:
            return a
        raise ValueError("cannot combine lattices")

    def __le__(self, other):
        """Test if one lattice includes another.

        Example:
            >>> Lattice(3) <= Lattice([1, 2, 3])
            True

        Args:
            other (Lattice): Lattice to compare with.

        Returns:
            bool
        """
        try:
            lat = Lattice(other, self.alignment)
        except ValueError:
            return False
        return lat == self

    def isdisjoint(self, other):
        """Test if lattices are disjoint.

        Lattices are considered disjoint if their alignments are disjoint.

        Example:
            >>> Lattice([1, 2, 3]).isdisjoint(Lattice(1))
            False

        Args:
            other (Lattice): Lattice to compare with.

        Returns:
            bool
        """
        for c in other.alignment:
            if c in self.alignment:
                return False
        return True


class WaveVector(collections.abc.Sequence):
    def __init__(self, seq=(), alignment=None):
        try:
            length = len(seq)
        except TypeError:
            length = 1
            seq = [seq]
        if length == 3:
            self._vec = tuple(seq)
        elif length == 2:
            if alignment in ("xy", None):
                self._vec = (seq[0], seq[1], np.nan)
            elif alignment == "yz":
                self._vec = (np.nan, seq[0], seq[1])
            elif alignment == "zx":
                self._vec = (seq[1], np.nan, seq[0])
            else:
                raise ValueError(f"invalid alignment: {alignment}")
        elif length == 1:
            if alignment in ("z", None):
                self._vec = (np.nan, np.nan, seq[0])
            elif alignment == "y":
                self._vec = (np.nan, seq[0], np.nan)
            elif alignment == "x":
                self._vec = (seq[0], np.nan, np.nan)
            else:
                raise ValueError(f"invalid alignment: {alignment}")
        elif length == 0:
            self._vec = (np.nan,) * 3
        else:
            raise ValueError("invalid sequence")

    def __str__(self):
        return str(self._vec)

    def __repr__(self):
        return self.__class__.__name__ + str(self._vec)

    def __eq__(self, other):
        other = WaveVector(other)
        for a, b in zip(self, other):
            if a != b and not (np.isnan(a) and np.isnan(b)):
                return False
        return True

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return self._vec[key]

    def __or__(self, other):
        other = WaveVector(other)
        seq = ()
        for a, b in zip(self, other):
            isnan = np.isnan(a) or np.isnan(b)
            if a != b and not isnan:
                raise ValueError("non-matching WaveVector")
            seq += (np.nan,) if isnan else (a,)
        return WaveVector(seq)

    def __and__(self, other):
        other = WaveVector(other)
        seq = ()
        for a, b in zip(self, other):
            if a != b and not (np.isnan(a) or np.isnan(b)):
                raise ValueError("non-matching WaveVector")
            seq += (b,) if np.isnan(a) else (a,)
        return WaveVector(seq)

    def permute(self, n=1):
        x, y, z = self
        n = n % 3
        for _ in range(n):
            x, y, z = z, x, y
        return WaveVector((x, y, z))

    def __le__(self, other):
        other = WaveVector(other)
        for a, b in zip(self, other):
            if a != b and not np.isnan(b):
                return False
        return True

    def isdisjoint(self, other):
        other = WaveVector(other)
        for a, b in zip(self, other):
            if not (np.isnan(a) or np.isnan(b)):
                return False
        return True
