---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  formats: md:myst
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

# Photonic Crystal

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell}
import treams
```

```{code-cell}
k0s = 2 * np.pi * np.linspace(0.01, 1, 200)
materials = [treams.Material(12.4), treams.Material()]
lmax = 3
radius = 0.2
lattice = treams.Lattice.cubic(0.5)
kpar = [0, 0, 0]
```

```{code-cell}
res = []
for k0 in k0s:
    sphere = treams.TMatrix.sphere(lmax, k0, radius, materials)
    svd = np.linalg.svd(sphere.latticeinteraction(lattice, kpar), compute_uv=False)
    res.append(svd[-1])
```

```{code-cell}
fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Smallest singular value")
ax.semilogy(299.792458 * k0s / (2 * np.pi), res)
fig.show()
```
