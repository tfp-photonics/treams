from ptsa import special as sc
from ptsa import _basisset

class Field:
    def efield(self, r):
        raise NotImplementedError
    def hfield(self, r):
        raise NotImplementedError
    def dfield(self, r):
        raise NotImplementedError
    def bfield(self, r):
        raise NotImplementedError
    def gfield(self, r, pol):
        raise NotImplementedError

class PlaneWave(Field):
    def __init__(self, kx, ky, kz, pol, coeff=None, helicity=True, k0=None, epsilon=None, mu=None, kappa=None):
        # Todo: test
        self.kx = np.atleast_1d(kx)
        self.ky = np.atleast_1d(ky)
        self.kz = np.atleast_1d(kz)
        self.pol = np.atleast_1d(pol)
        self.k0 = k0
        self.epsilon = epsilon
        self.mu = mu
        self.kappa = kappa
        self.helicity = helicity
        self.coeff = coeff
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

    def field(r):
        r = np.array(r)
        if r.ndim == 1:
            r = np.reshape(r, (1, -1))
        if self.helicity:
            return sc.vpw_A(
                self.kx,
                self.ky,
                self.kz,
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
                self.pol,
            )
        else:
            return (1 - self.pol[:, None]) * sc.vpw_M(
                self.kx,
                self.ky,
                self.kz,
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            ) + self.pol[:, None] * sc.vpw_N(
                self.kx,
                self.ky,
                self.kz[choice, :],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            )

class SphericalWave(Field, _basisset.SphericalWaveBasis):
    def __init__(self, coeff, pidx, l, m, pol=None, /, k0=None, epsilon=1, mu=1, kappa=1, positions=None, helicity=True, wavetype=None):
        super().__init__(pidx, l, m, pol, positions, helicity, wavetype)
        self.epsilon = epsilon
        self.mu = mu
        self.kappa = kappa
        coeff = np.atleast_1d(coeff)
        if (
                coeff.ndim > 2
                or (coeff.ndim in (1, 2) and coeff.shape[0] != self.l.shape[0])
        ):
            raise ValueError(f"invalid shape of coeff '{coeff.shape}'")
        self.coeff = coeff
        self.k0 = k0

    @classmethod
    def from(cls, source, basis):
        pidx, l, m, pol = modes
        positions = np.zeros((3, 1)) if positions is None else positions
        if positions.ndim == 1:
            positions = positions[:, None]
        elif positions.ndim != 2:
            raise ValueError
        pos = (*(i[:, None] for i in positions[pidx, :].T),)
        if isinstance(source, PlaneWave):
            m = pw.to_sw(
                l, m, pol, source.kx, source.ky, source.kz, source.pol, source.helicity
            ) * pw.translate(source.kx, source.ky, source.kz, *pos)
        elif isinstance(source, CylindricalWave):
            m = cw.to_sw(
                *(m[:, None] for m in self.modes),
                source.kz,
                source.m,
                source.pol,
                self.ks[pol],
                posout=self.pidx[:, None],
                posin=pidx,
                helicity=self.helicity,
            )
        elif isinstance(source, SphericalWave):
            pass
        else:
            raise ValueError

    def pick(self, modes):
        pass

    def field(r):
        r = np.array(r)
        if r.ndim == 1:
            r = np.reshape(r, (1, -1))
        r_sph = sc.car2sph(r[..., None, :] - self.positions)
        if scattered:
            wave_A = sc.vsw_A
            wave_M = sc.vsw_M
            wave_N = sc.vsw_N
        else:
            wave_A = sc.vsw_rA
            wave_M = sc.vsw_rM
            wave_N = sc.vsw_rN
        if self.helicity:
            res = wave_A(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
                self.pol,
            )
        else:
            res = (1 - self.pol[:, None]) * wave_M(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
            ) + self.pol[:, None] * wave_N(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
            )
        return sc.vsph2car(res, r_sph[..., self.pidx, :])

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


    def field(r):
        r = np.array(r)
        if r.ndim == 1:
            r = np.reshape(r, (1, -1))
        r_cyl = sc.car2cyl(r[..., None, :] - self.positions)
        if self.scattered:
            wave_A = sc.vcw_A
            wave_M = sc.vcw_M
            wave_N = sc.vcw_N
        else:
            wave_A = sc.vcw_rA
            wave_M = sc.vcw_rM
            wave_N = sc.vcw_rN
        if self.helicity:
            res = wave_A(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
                self.ks[self.pol],
                self.pol,
            )
        else:
            res = (1 - self.pol[:, None]) * wave_M(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
            ) + self.pol[:, None] * wave_N(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
                self.ks[self.pol],
            )
        return sc.vcyl2car(res, r_cyl[..., self.pidx, :])
