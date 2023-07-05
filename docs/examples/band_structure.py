import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(1 / 10_000, 1 / 350, 200)
material_slab = 3
thickness = 10
material_sphere = (4 + 0.1j, 1, 0.05)
lattice = treams.Lattice.square(500)
radius = 100
lmax = 3
az = 210

res = []
for i, k0 in enumerate(k0s):
    kpar = [0, 0]
    spheres = treams.TMatrix.sphere(
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(lattice, kpar)

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(spheres, pwb)
    total = treams.SMatrices.stack([slab, dist, array, dist])
    x, _ = total.bands_kz(az)
    x = x * az / np.pi
    sel = x[np.abs(np.imag(x)) < 0.1]
    res.append(sel)


fig, ax = plt.subplots()
for k0, sel in zip(k0s, res):
    ax.scatter(sel.real, len(sel) * [299792.458 * k0 / (2 * np.pi)], 0.2, c="C0")
    ax.scatter(sel.imag, len(sel) * [299792.458 * k0 / (2 * np.pi)], 0.2, c="C1")
ax.set_xlabel("$k_z a_z / \\pi$")
ax.set_ylabel("Frequency (THz)")
ax.set_xlim([-1, 1])
ax.set_ylim(ymin=0)
ax.legend(["$Real$", "$Imag$"])
fig.show()
