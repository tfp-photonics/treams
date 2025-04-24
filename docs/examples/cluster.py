import matplotlib.pyplot as plt
import numpy as np

import treams

k0 = 2 * np.pi / 1000
materials = [treams.Material(16 + 0.5j), treams.Material()]
lmax = 3
radii = [110, 90, 80, 75]
positions = (220 / np.sqrt(24)) * np.array(
    [
        [np.sqrt(8), 0, -1],
        [-np.sqrt(2), np.sqrt(6), -1],
        [-np.sqrt(2), -np.sqrt(6), -1],
        [0, 0, 3],
    ]
)

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
tm = treams.TMatrix.cluster(spheres, positions).interaction.solve()

inc = treams.plane_wave([k0, 0, 0], 0, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)
xs = tm.xs(inc)

grid = np.mgrid[-300:300:101j, 0:1, -300:300:101j].squeeze().transpose((1, 2, 0))
intensity = np.zeros_like(grid[..., 0])
valid = tm.valid_points(grid, radii)
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0], grid[0, :, 2], intensity.T, shading="nearest", vmin=0, vmax=2,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
ax.text(
    0,
    0,
    r"$\sigma_\mathrm{sca} = "
    f"{xs[0]:.7}\\,$nm$^2$\n"
    r"$\sigma_\mathrm{ext} = "
    f"{xs[1]:.7}\\,$nm$^2$",
)
fig.show()

tm_global = tm.expand(treams.SphericalWaveBasis.default(6))
sca = tm_global @ inc.expand(tm_global.basis)

xs = tm_global.xs(inc)

grid = np.mgrid[-300:300:101j, 0:1, -300:300:101j].squeeze().transpose((1, 2, 0))
intensity_global = np.zeros_like(grid[..., 0])
valid = tm_global.valid_points(grid, [260])
intensity_global[~valid] = np.nan
intensity_global[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0], grid[0, :, 2], intensity_global.T, shading="nearest", vmin=0, vmax=2,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
ax.text(
    0,
    0,
    r"$\sigma_\mathrm{sca} = "
    f"{xs[0]:.7}\\,$nm$^2$\n"
    r"$\sigma_\mathrm{ext} = "
    f"{xs[1]:.7}\\,$nm$^2$",
)
fig.show()

inc = treams.plane_wave([0, 0, -k0], 0, k0=tm.k0, material=tm.material)
tm_rotate = tm_global.rotate(0, np.pi / 2)
sca = tm_rotate @ inc.expand(tm_rotate.basis)

xs = tm_rotate.xs(inc)

grid = np.mgrid[-300:300:101j, 0:1, -300:300:101j].squeeze().transpose((1, 2, 0))
intensity_global = np.zeros_like(grid[..., 0])
valid = tm_rotate.valid_points(grid, [260])
intensity_global[~valid] = np.nan
intensity_global[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0], grid[0, :, 2], intensity_global.T, shading="nearest", vmin=0, vmax=2,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
ax.text(
    0,
    0,
    r"$\sigma_\mathrm{sca} = "
    f"{xs[0]:.7}\\,$nm$^2$\n"
    r"$\sigma_\mathrm{ext} = "
    f"{xs[1]:.7}\\,$nm$^2$",
)
fig.show()
