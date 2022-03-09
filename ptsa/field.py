from ptsa import special as sc

class Field:
    def field(self, modes):
        NotImplementedError
    def hfield(self, modes):
        NotImplementedError
    def poynting(self, modes):
        NotImplementedError

class PlaneWave(Field):
    def __init__(self, kx, ky, kz, pol, coeff=None, helicity=True, k0=None, epsilon=None, mu=None, kappa=None):
        self.kx = np.atleast_1d(kx)
        self.ky = np.atleast_1d(ky)
        self.kz = np.atleast_1d(kz)
        self.pol = np.atleast_1d(pol)
        self.k0 = k0
        self.epsilon = epsilon
        self.mu = mu
        self.kappa = kappa
        self.helicity = helicity
        self.coeff = np.zeros_like(kx, complex) if coeff is None else coeff
        fix = ''.join(fix.lower().split())
        self._fix = fix

    @classmethod
    def by_efield(cls, kx, ky, kz, pol, evec, helicity=True, k0=None, epsilon=None, mu=None, kappa=None):
        if helicity:
            v = sc.vpw_A(kx, ky, kz, 0, 0, 0, 1 - pol)
        else:
            v = (1 - pol) * sc.vpw_N(kx, ky, kz, 0, 0, 0) - pol * sc.vpw_M(kx, ky, kz, 0, 0, 0)
        coeff = np.sum(evec * v, axis=-1)
        return cls(kx, ky, kz, pol, coeff, helicity, k0, epsilon, mu, kappa)

    @classmethod
    def by_diffr_orders_circle(cls, kx, ky, pol, ks, a, rmax, helicity=True):
        pass

    @classmethod
    def by_diffr_orders_grid(cls, kx, ky, pol, ks, a, n, helicity=True):
        pass

    @classmethod
    def from(cls, source, kx, ky, kz, pol):
        if isinstance(source, PlaneWave):
            pass
        elif isinstance(source, SphericalWave):
            pass
        elif isinstance(source, CylindricalWave):
            pass
        else:
            raise ValueError

    @property
    def modes(self):
        """
        Modes of the Q-matrix

        X- and Y-components and polarization of each row/column of the Q-matrix

        Returns:
            3-tuple
        """
        return self.kx, self.ky, self.pol

    def translate(self, modes):
        pass

    def flip(self, modes):
        pass

    def pick(self, modes):
        pass

class SphericalWave(Field):
    def __init__(self, k0, l, m, pol, epsilon=1, mu=1, kappa=1, coeff=None, helicity=True):
        self.k0, self.l, self.m, self.pol =
        self.epsilon = self.kappa = self.mu = None
        self.helicity = helicity
        coeff=np.zeros_like(kx, complex)

    @classmethod
    def from(cls, source, modes, positions=None):
        pidx, l, m, pol = modes
        pos = (*(i[:, None] for i in positions[pidx, :].T),)
        if isinstance(source, PlaneWave):
            m = pw.to_sw(
                l, m, pol, source.kx, source.ky, source.kz, source.pol, source.helicity
                ) * pw.translate(source.kx, source.ky, source.kz)
        elif isinstance(source, SphericalWave):
            pass
        elif isinstance(source, CylindricalWave):
            m = cw.to_sw(

            )
        else:
            raise ValueError

    def pick(self, modes):
        pass

class CylindricalWave(Field):
    def __init__(self, k0, l, m, pol, epsilon=1, mu=1, kappa=1, coeff=None, helicity=True):
        self.k0, self.l, self.m, self.pol =
        self.epsilon = self.kappa = self.mu = None
        self.helicity = helicity
        coeff=np.zeros_like(kx, complex)

    @classmethod
    def from(cls, source, kx, ky, kz, pol):
        if isinstance(source, PlaneWave):
            pass
        elif isinstance(source, SphericalWave):
            pass
        elif isinstance(source, CylindricalWave):
            pass
        else:
            raise ValueError

    def pick(self, modes):
        pass
