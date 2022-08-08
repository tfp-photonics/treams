import numpy as np

import ptsa.lattice as la


class Lattice:
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
        if isinstance(arr, Lattice):
            self._alignment = arr.alignment if alignment is None else alignment
            self._lattice = arr._sublattice(self._alignment)[...]
            return

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
        return self._alignment

    @classmethod
    def square(cls, pitch, alignment=None):
        return cls(2 * [pitch], alignment)

    @classmethod
    def cubic(cls, pitch, alignment=None):
        return cls(3 * [pitch], alignment)

    @classmethod
    def rectangular(cls, x, y, alignment=None):
        return cls([x, y], alignment)

    @classmethod
    def orthorhombic(cls, x, y, z, alignment=None):
        return cls([x, y, z], alignment)

    @classmethod
    def hexagonal(cls, pitch, height=None, alignment=None):
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
        return self is other or (
            self.alignment == other.alignment
            and self.dim == other.dim
            and np.all(self[...] == other[...])
        )

    @property
    def dim(self):
        if self._lattice.ndim == 0:
            return 1
        return self._lattice.shape[0]

    @property
    def volume(self):
        if self.dim == 1:
            return self._lattice
        return la.volume(self._lattice)

    @property
    def reciprocal(self):
        return self._reciprocal

    def __getitem__(self, idx):
        return self._lattice[idx]

    def __str__(self):
        return str(self._lattice)

    def __repr__(self):
        string = str(self._lattice).replace("\n", "\n        ")
        return f"Lattice({string}, alignment='{self.alignment}')"

    def _sublattice(self, key):
        key = key.lower()
        if self.dim == 1:
            if key == self.alignment:
                return Lattice(self)
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
        return True

    def __or__(self, other):
        if self == other:
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
            if self.alignment == "x":
                arr[0, 0] = self[...]
                arr[1:, 1:] = other[...]
                return Lattice(arr)
            if self.alignment == "y":
                arr[1, 1] = self[...]
                arr[[[2], [0]], [2, 0]] = other[...]
                return Lattice(arr)
            if self.alignment == "z":
                arr[2, 2] = self[...]
                arr[:2, :2] = other[...]
                return Lattice(arr)
            if other.alignment == "x":
                arr[0, 0] = other[...]
                arr[1:, 1:] = self[...]
                return Lattice(arr)
            if other.alignment == "y":
                arr[1, 1] = other[...]
                arr[[[2], [0]], [2, 0]] = self[...]
                return Lattice(arr)
            if other.alignment == "z":
                arr[2, 2] = other[...]
                arr[:2, :2] = self[...]
                return Lattice(arr)
        raise ValueError("cannot combine lattices")

    def __and__(self, other):
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
        try:
            lat = Lattice(other, self.alignment)
        except ValueError:
            return False
        return lat == self

