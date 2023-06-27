import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(1 / 700, 1 / 300, 200)
materials = [treams.Material(16 + 0.5j), treams.Material()]
lmax = 4
radius = 75
spheres = [treams.TMatrix.sphere(lmax, k0, radius, materials) for k0 in k0s]

xs_sca = np.array([tm.xs_sca_avg for tm in spheres]) / (np.pi * radius**2)
xs_ext = np.array([tm.xs_ext_avg for tm in spheres]) / (np.pi * radius**2)

swb_lmax1 = treams.SphericalWaveBasis.default(1)
spheres_lmax1 = [tm[swb_lmax1] for tm in spheres]
xs_sca_lmax1 = np.array([tm.xs_sca_avg for tm in spheres_lmax1]) / (np.pi * radius**2)
xs_ext_lmax1 = np.array([tm.xs_ext_avg for tm in spheres_lmax1]) / (np.pi * radius**2)

tm = spheres[-1]
inc = treams.plane_wave([0, 0, tm.k0], 1, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)
grid = np.mgrid[-100:100:101j, 0:1, -100:100:101j].squeeze().transpose((1, 2, 0))
intensity = np.zeros_like(grid[..., 0])
valid = tm.valid_points(grid, [radius])
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

fig, ax = plt.subplots()
ax.plot(k0s * 299792.458 / (2 * np.pi), xs_ext)
ax.plot(k0s * 299792.458 / (2 * np.pi), xs_sca)
ax.plot(k0s * 299792.458 / (2 * np.pi), xs_ext_lmax1, color="C0", linestyle=":")
ax.plot(k0s * 299792.458 / (2 * np.pi), xs_sca_lmax1, color="C1", linestyle=":")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Efficiency")
ax.legend(["Extinction", "Scattering", "Extinction $l=1$", "Scattering $l=1$"])
fig.show()

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[0, :, 2],
    grid[:, 0, 0],
    intensity.T,
    shading="nearest",
    vmin=0,
    vmax=1,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
fig.show()
