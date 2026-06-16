# Tutorial: PL Spectrum — Force Mode

**Use this mode when you have** vertical forces from single-point DFT calculations on fixed
geometries rather than two independently relaxed structures. This approach is favored in
high-throughput screening because it avoids converging an excited-state relaxation.

---

## 1. Physics background

At the ground-state equilibrium geometry $Q_g$, the force difference between the excited-state
and ground-state electronic structures encodes the same coupling information as the structural
displacement [Alkauskas *et al.*, *New J. Phys.* **16**, 073026 (2014), Eq. 7]:

$$
q_k = \frac{1}{\omega_k^2} \sum_{a,i} \frac{F_{e,a,i} - F_{g,a,i}}{\sqrt{m_a}}\, e_{k,a,i}
$$

The approximation is exact under the harmonic potential; deviations indicate anharmonicity.

---

## 2. DFT setup

| Job | Description | Key INCAR settings |
|-----|-------------|-------------------|
| `gs/` | Ground-state single point at $Q_g$ | `NSW = 1; IBRION = -1; ISPIN = 2` |
| `es/` | Excited-state single point at $Q_g$ | As above; manually constrain excited occupation |
| `phonon/` | Gamma-point phonons | `IBRION = 8; NSW = 1` |

Both `gs/` and `es/` must use **identical atomic positions** (copy the ground-state POSCAR).

---

## 3. Compute the PL lineshape

### CLI

```bash
defectpl pl force \
    --band_yaml   phonon/band.yaml \
    --outcar_gs   gs/OUTCAR \
    --outcar_es   es/OUTCAR \
    --ezpl        1.945 \
    --gamma       2.0 \
    --json_out    pl_force.json \
    --plot_all
```

### Python API

```python
from defectpl.phonon import read_band_yaml
from defectpl.vasp_wrapper import prepare_dF_files
from defectpl.defectpl import Photoluminescence
from monty.serialization import dumpfn

frequencies, eigenvectors, masses = read_band_yaml("phonon/band.yaml")
dF = prepare_dF_files("gs/OUTCAR", "es/OUTCAR")   # shape (natoms, 3) in eV/Å

pl = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    EZPL=1.945,
    dR=None,
    dF=dF,
    gamma=2.0,
)

print(f"S = {pl.HR_factor:.3f}")
pl.generate_plots(out_dir="plots/", fig_format="png")
dumpfn(pl, "pl_force.json", indent=4)
```

---

## 4. Note on ΔQ and ΔR

In force mode, the actual structural displacement $\Delta Q$ and $\Delta R$ are not computed
(they require the excited-state geometry). The corresponding fields in the output JSON are set
to `0.0`. All Huang–Rhys quantities ($S_k$, $S$, $S(\omega)$) remain valid.

---

## 5. Checking consistency with displacement mode

If you have access to both modes, compare the resulting $S_k$ values:

```python
import json, numpy as np

with open("pl.json") as f:     d1 = json.load(f)
with open("pl_force.json") as f: d2 = json.load(f)

Sks_disp  = np.array(d1["Sks"])
Sks_force = np.array(d2["Sks"])

print(f"S (displacement) = {Sks_disp.sum():.4f}")
print(f"S (force mode)   = {Sks_force.sum():.4f}")
```

Agreement within ~5% indicates a well-converged harmonic approximation.
