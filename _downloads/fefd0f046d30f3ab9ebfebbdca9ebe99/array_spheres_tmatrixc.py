import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, 100)
material_slab = 3
thickness = 10
material_sphere = (4, 1, 0.05)
pitch = 500
lattice = treams.Lattice.square(pitch)
radius = 100
lmax = mmax = 3

tr = np.zeros((len(k0s), 2))
for i, k0 in enumerate(k0s):
    kpar = [0, 0.3 * k0]

    spheres = treams.TMatrix.sphere(
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(pitch, kpar[0])
    cwb = treams.CylindricalWaveBasis.diffr_orders(kpar[0], mmax, pitch, 0.02)
    spheres_cw = treams.TMatrixC.from_array(spheres, cwb)
    chain_cw = spheres_cw.latticeinteraction.solve(pitch, kpar[1])

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(chain_cw, pwb)
    total = treams.SMatrices.stack([slab, dist, array])
    tr[i, :] = total.tr(plw)

fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 0])
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 1])
ax.legend(["$T$", "$R$"])
fig.show()
