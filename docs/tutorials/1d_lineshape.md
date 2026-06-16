# Tutorial: 1D Analytical Lineshape

The `VibrationalSpectra1D` class implements a **single-mode displaced and distorted harmonic
oscillator** lineshape. It does not require phonon calculations — only four scalar parameters —
and is useful for:

- Quick validation against experiment
- Defects dominated by a single effective phonon mode
- Temperature-dependent spectra

---

## 1. Theory recap

The 1D model allows $\omega_1 \neq \omega_2$ (ground vs excited frequency), unlike the
multi-mode generating-function approach which assumes identical Hessians. The Franck–Condon
overlap elements $M_{i,j} = \langle \chi_{g,j} | \chi_{e,i} \rangle$ are computed analytically
using Hermite polynomials.

At finite temperature $T$, the lineshape is:

$$
L(E) \propto E^3 \sum_{i,j} W_i |M_{i,j}|^2 \, g(E - E_{i,j})
$$

where $W_i = e^{-i\hbar\omega_1/k_BT}/Z$ is the Boltzmann weight for the $i$-th excited-state
vibrational level, $E_{i,j} = E_\text{ZPL} - j\hbar\omega_2 + i\hbar\omega_1$, and $g$ is a
Gaussian broadening kernel.

---

## 2. CLI

```bash
defectpl spectra1d \
    --ezpl    1.945 \
    --w1      38.5 \
    --w2      42.0 \
    --dq_val  1.35 \
    --temp    300.0 \
    --points  5000 \
    --plot \
    --save_prefix nv_center_1d
```

| Option | Description |
|--------|-------------|
| `--ezpl` | Zero-phonon line energy in eV (required) |
| `--w1` | Ground-state effective frequency in meV (required) |
| `--w2` | Excited-state effective frequency in meV (required) |
| `--dq_val` | Configuration coordinate $\Delta Q$ in $\sqrt{\text{amu}}\cdot\text{Å}$ (required) |
| `--temp` | Temperature in K (default: 300) |
| `--points` | Number of energy grid points (default: 5000) |
| `--plot` | Show/save the lineshape plot |
| `--save_prefix` | Prefix for output JSON files |

---

## 3. Python API

```python
from defectpl.defectpl import VibrationalSpectra1D

spec = VibrationalSpectra1D(
    EZPL   = 1.945,   # eV
    w1_meV = 38.5,    # ground-state frequency  (meV)
    w2_meV = 42.0,    # excited-state frequency (meV)
    DQ     = 1.35,    # amu^0.5 · Å
    T      = 300.0,   # K
    E0     = 1.5,     # eV — lower bound of energy grid
    dE     = 5e-4,    # eV — grid spacing
    M      = 30,      # number of vibrational levels per manifold
)

spec.compute_overlap_matrix()
spec.compute_spectrum()
spec.compute_lineshape()

print(f"ZPL peak position : {spec.get_peak_position():.4f} eV")
print(f"FWHM              : {spec.get_fwhm()*1000:.1f} meV")

spec.plot_lineshape(out="lineshape_1d.pdf")
```

---

## 4. Getting $\Delta Q$, $\omega_1$, $\omega_2$ from first principles

| Quantity | How to obtain |
|----------|--------------|
| $\Delta Q$ | `defectpl dq CONTCAR_gs CONTCAR_es` |
| $\omega_1$ | Fit ground-state CCD parabola (`defectpl analyze-ccd`) |
| $\omega_2$ | Fit excited-state CCD parabola |

Or from the multi-mode spectrum:

```python
import numpy as np

# After running Photoluminescence
omega_eff = np.sum(pl.Sks * pl.frequencies) / np.sum(pl.Sks)  # in eV
print(f"ω_eff = {omega_eff * 1000:.1f} meV")
```

---

## 5. Temperature dependence

Compare T = 0 K and T = 300 K:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for T, ls in [(0, "-"), (100, "--"), (300, ":")]:
    spec = VibrationalSpectra1D(EZPL=1.945, w1_meV=38.5, w2_meV=42.0,
                                DQ=1.35, T=T)
    spec.compute_overlap_matrix()
    spec.compute_spectrum()
    spec.compute_lineshape()
    ax.plot(spec.energies, spec.lineshape, ls=ls, label=f"{T} K")

ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Intensity (arb. units)")
ax.legend()
fig.savefig("temperature_series.pdf", dpi=150)
```
