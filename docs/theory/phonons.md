# Phonon Calculations

## Harmonic approximation

Within the **harmonic approximation**, the total potential energy of a periodic solid is expanded
to second order in atomic displacements $\{u_{a,i}\}$ about the equilibrium configuration:

$$
V \approx V_0 + \frac{1}{2}\sum_{a,i}\sum_{b,j} \Phi_{ai,bj}\, u_{a,i}\, u_{b,j}
$$

The **interatomic force constants (IFCs)** are defined as:

$$
\Phi_{ai,bj} = \frac{\partial^2 V}{\partial u_{a,i}\,\partial u_{b,j}}\Bigg|_{\mathbf{u}=0}
$$

where $a,b$ index atoms and $i,j \in \{x,y,z\}$ label Cartesian directions.

## Dynamical matrix and normal modes

The **dynamical matrix** at wavevector $\mathbf{q}$ is:

$$
D_{ai,bj}(\mathbf{q}) = \frac{1}{\sqrt{m_a m_b}}
\sum_{\mathbf{R}} \Phi_{ai,bj}(\mathbf{R})\, e^{i\mathbf{q}\cdot\mathbf{R}}
$$

Diagonalizing $D(\mathbf{q})$ gives $3N$ phonon branches with frequencies $\omega_{k}(\mathbf{q})$
and normalized polarization vectors $\mathbf{e}_{k}(\mathbf{q})$:

$$
\sum_{b,j} D_{ai,bj}(\mathbf{q})\, e_{k,b,j}(\mathbf{q}) = \omega_k^2(\mathbf{q})\, e_{k,a,i}(\mathbf{q})
$$

For point-defect lineshape calculations, DefectPL uses only **Gamma-point** ($\mathbf{q}=0$)
phonons of a large defect supercell, so the $\mathbf{q}$ index is dropped throughout.

## Eigenvector normalization

Polarization vectors returned by phonopy satisfy:

$$
\sum_{a,i} e_{k,a,i}^2 = 1, \qquad \sum_{a,i} e_{k,a,i}\, e_{k',a,i} = \delta_{kk'}
$$

DefectPL stores them in the shape `(nmodes, natoms, 3)`, consistent with what `read_band_yaml`
returns.

## Phonon Inverse Participation Ratio

The **phonon IPR** quantifies how spatially confined a vibrational mode is
[Alkauskas *et al.*, *New J. Phys.* **16**, 073026 (2014)].
The per-atom weight of mode $k$ on atom $a$ is:

$$
p_{k,a} = \sum_i e_{k,a,i}^2
$$

and the IPR is:

$$
\text{IPR}_k = \frac{\sum_a p_{k,a}^2}{\left(\sum_a p_{k,a}\right)^2}
$$

| IPR | Mode character |
|-----|---------------|
| $1/N$ | Perfectly delocalized over $N$ atoms (acoustic/bulk-like) |
| $1$ | Fully localized on a single atom (local vibrational mode) |

The **localization ratio** $\beta_k = N \cdot \text{IPR}_k$ provides an atom-count-normalized
index: $\beta = 1$ for a bulk mode; $\beta = N$ for a fully localized mode.

!!! note "Historical note"
    In the NV center in diamond, the dominant 65 meV mode that couples strongly to the optical
    transition shows $\beta \approx 11$ in a 512-atom supercell, indicating it is a quasi-local
    resonance (partially confined) rather than a true local mode ($\beta \to \infty$).

## Frequency unit conversion

phonopy outputs frequencies in THz. DefectPL converts to eV using:

$$
1\,\text{THz} = 4.13567\times10^{-3}\,\text{eV}
$$

(from the constant `THZ2EV` in `defectpl.constants`).

## Required VASP settings

**DFPT method (recommended for LDA/GGA):**

```
IBRION = 8    # density-functional perturbation theory
NSW    = 1
ENCUT  = <converged value>
```

**Finite-displacement method (required for hybrid functionals):**

Use `phonopy` to generate displaced supercells, run each with `IBRION = -1; NSW = 0`, collect
forces, then build force constants with `phonopy --force-sets`.

## DefectPL workflow

```python
from defectpl.phonon import (
    create_force_constants_from_vasprun,
    calculate_gamma_phonon_to_band_yaml,
    read_band_yaml,
    extract_gamma_phonon_data,
)

# 1. Extract IFCs from DFPT output
create_force_constants_from_vasprun("vasprun.xml")

# 2. Compute Gamma-point band.yaml
calculate_gamma_phonon_to_band_yaml(
    unitcell_filename="POSCAR",
    force_constants_filename="FORCE_CONSTANTS",
    dimension="2 2 2",
    output_filename="band.yaml",
)

# 3a. Low-level: returns (frequencies, eigenvectors, masses) arrays
frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

# 3b. High-level: returns a GammaPhononData object with .as_dict() support
phonon_data = extract_gamma_phonon_data("band.yaml")
```

CLI equivalents:

```bash
defectpl phonon-fc vasprun.xml
defectpl phonon-band --poscar POSCAR --fc FORCE_CONSTANTS --dim "2 2 2" --out band.yaml
defectpl phonon-parse band.yaml --json_out phonon_data.json
defectpl phonon-symm --poscar POSCAR --fc FORCE_CONSTANTS --dim "2 2 2"
```
