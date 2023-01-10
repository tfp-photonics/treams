import numpy as np

from ptsa import misc


class Material:
    def __init__(self, epsilon=1, mu=1, kappa=0):
        if isinstance(epsilon, Material):
            epsilon, mu, kappa = epsilon()
        elif isinstance(epsilon, (tuple, list, np.ndarray)):
            if len(epsilon) == 1:
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
        return self._epsilon

    @property
    def mu(self):
        return self._mu

    @property
    def kappa(self):
        return self._kappa

    def __iter__(self):
        return iter((self.epsilon, self.mu, self.kappa))

    @classmethod
    def from_n(cls, n=1, impedance=1, kappa=0):
        epsilon = n / impedance
        mu = n * impedance
        return cls(epsilon, mu, kappa)

    @classmethod
    def from_nmp(cls, ns=None, impedance=1):
        if ns is None:
            ns = (1, 1)
        epsilon = sum(ns) * 0.5 / impedance
        mu = sum(ns) * 0.5 * impedance
        kappa = (ns[1] - ns[0]) * 0.5
        return cls(epsilon, mu, kappa)

    @property
    def n(self):
        if self.kappa == 0:
            n = np.sqrt(self.epsilon * self.mu)
            if n.imag < 0:
                n = -n
            return n
        raise ValueError("'n' is not unique in chiral media, consider using 'nmp'")

    @property
    def nmp(self):
        return misc.refractive_index(self.epsilon, self.mu, self.kappa)

    @property
    def impedance(self):
        return np.sqrt(self.mu / self.epsilon)

    def __call__(self):
        return self.epsilon, self.mu, self.kappa

    def __eq__(self, other):
        if not isinstance(other, Material):
            other = Material(*other)
        return (
            super().__eq__(other)
            and self.epsilon == other.epsilon
            and self.mu == other.mu
            and self.kappa == other.kappa
        )

    @property
    def ischiral(self):
        return self.kappa != 0

    def __str__(self):
        return "(" + ", ".join([str(i) for i in self()]) + ")"

    def __repr__(self):
        return self.__class__.__name__ + str(self)


vacuum = Material()
