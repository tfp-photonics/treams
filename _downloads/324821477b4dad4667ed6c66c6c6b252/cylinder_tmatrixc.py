import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(1 / 6000, 1 / 300, 200)
materials = [treams.Material(16 + 0.5j), treams.Material()]
mmax = 4
radius = 75
kzs = [0]
cylinders = [treams.TMatrixC.cylinder(kzs, mmax, k0, radius, materials) for k0 in k0s]

xw_sca = np.array([tm.xw_sca_avg for tm in cylinders]) / (2 * radius)
xw_ext = np.array([tm.xw_ext_avg for tm in cylinders]) / (2 * radius)

cwb_mmax0 = treams.CylindricalWaveBasis.default(kzs, 0)
cylinders_mmax0 = [tm[cwb_mmax0] for tm in cylinders]
xw_sca_mmax0 = np.array([tm.xw_sca_avg for tm in cylinders_mmax0]) / (2 * radius)
xw_ext_mmax0 = np.array([tm.xw_ext_avg for tm in cylinders_mmax0]) / (2 * radius)

tm = cylinders[-1]
inc = treams.plane_wave([tm.k0, 0, 0], 1, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)
grid = np.mgrid[-100:100:101j, -100:100:101j, 0:1].squeeze().transpose((1, 2, 0))
intensity = np.zeros_like(grid[..., 0])
valid = tm.valid_points(grid, [radius])
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

fig, ax = plt.subplots()
ax.plot(k0s * 299792.458 / (2 * np.pi), xw_ext)
ax.plot(k0s * 299792.458 / (2 * np.pi), xw_sca)
ax.plot(k0s * 299792.458 / (2 * np.pi), xw_ext_mmax0, color="C0", linestyle=":")
ax.plot(k0s * 299792.458 / (2 * np.pi), xw_sca_mmax0, color="C1", linestyle=":")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Efficiency")
ax.legend(["Extinction", "Scattering", "Extinction $m=1$", "Scattering $m=1$"])
fig.show()

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[0, :, 1], grid[:, 0, 0], intensity.T, shading="nearest", vmin=0, vmax=1,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
fig.show()
