import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(1 / 1000, 1 / 300, 50)
materials = [(), (12.4 + 1j, 1 + 0.1j, 0.5 + 0.05j), (2, 2)]
thickness = 50
tr = np.zeros((2, len(k0s), 2))
for i, k0 in enumerate(k0s):
    pwb = treams.PlaneWaveBasisByComp.default([0, 0.5 * k0])
    slab = treams.SMatrices.slab(thickness, pwb, k0, materials)
    tr[0, i, :] = slab.tr([1, 0])
    tr[1, i, :] = slab.tr([0, 1])

fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.plot(299792.458 * k0s / (2 * np.pi), tr[0, :, 0])
ax.plot(299792.458 * k0s / (2 * np.pi), tr[0, :, 1])
ax.plot(299792.458 * k0s / (2 * np.pi), tr[1, :, 0], c="C0", linestyle="--")
ax.plot(299792.458 * k0s / (2 * np.pi), tr[1, :, 1], c="C1", linestyle="--")
ax.legend(["$T_+$", "$R_+$", "$T_-$", "$R_-$"])
fig.show()
