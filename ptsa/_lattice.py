import numpy as np

import ptsa.lattice as la

class Lattice:
    _allowed_alignments = {
        "x": "y", "y": "z", "z": "x", "xy": "yz", "yz": "zx", "zx": "xy", "xyz": "xyz"
    }
    def __init__(self, arr, alignment=None):
        if isinstance(arr, Lattice):
            self._alignment = arr.alignment if alignment is None else alignment
            self._lattice = arr.sublattice(self._alignment)
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

    def __eq__(self):
        return (
            self.alignment == other.alignment
            and self.dim == other.dim
            and np.all(self.lattice == other.lattice)
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

    def __len__(self):
        return len(self._lattice)

    def __str__(self):
        return str(self._lattice)

    def __repr__(self):
        string = str(self._lattice).replace("\n", "\n        ")
        return f"Lattice({string}, alignment='{self.alignment}')"

    def sublattice(self, key):
        key = key.lower()
        if self.dim == 1:
            if key == self.alignment:
                return Lattice(self)
            raise ValueError
        idx = []
        for c in key:
            idx.append(self.alignment.find(c))
        if -1 in idx or key not in self._allowed_alignments:
            raise ValueError("invalid key")
        idx_opp = [i for i in range(len(self)) if i not in idx]
        mask = np.any(self[:, idx] != 0, axis=-1)
        if (
            (self[mask, :][:, idx_opp] != 0).any()
            or (self[np.logical_not(mask), :][:, idx] != 0).any()
            or len(idx) != sum(mask)
        ):
            raise ValueError("cannot detemine sublattice")
        return Lattice(self[mask, :][:, idx], key)

    def permute(self, n=1):
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        lattice = self.lattice
        alignment = self.alignment
        if self.ndim == 3:
            while n > 0:
                lattice = lattice[:, 2, 0, 1]
        else:
            while n > 0:
                alignment = alignment[self._allowed_alignments[alignment]]
        return Lattice(lattice, alignment)

    def __bool__(self):
        return True
