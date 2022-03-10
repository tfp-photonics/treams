from ptsa import misc

class Material:
    def __init__(epsilon=1, mu=1, kappa=0):
        self.epsilon = epsilon
        self.mu = mu
        self.kappa = kappa

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
            if x.imag < 0:
                n = -n
            return n
        raise ValueError("'n' not unique in chiral media, consider using 'nmp'")

    @property
    def nmp(self):
        return misc.refractive_index(self.epsilon, self.mu, self.kappa)

    @property
    def impedance(self):
        return np.sqrt(self.mu / self.epsilon)

    @property
    def parameters(self):
        return self.epsilon, self.mu, self.kappa

    def __eq__(self, x):
        if not isinstance(x, Material):
            x = Material(*x)
        return self.epsilon == x.epsilon and self.mu == x.mu and self.kappa == x.kappa
