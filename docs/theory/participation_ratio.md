# Electronic State Localization

## Site-projected wavefunction character

For a Kohn–Sham state $|\psi_{n,\mathbf{k},\sigma}\rangle$, VASP decomposes the wavefunction onto
spherical harmonics centred at each lattice site $a$:

$$
p_a(n,\mathbf{k},\sigma) = \sum_{\ell m}
\left|\langle Y^{a}_{\ell m} \mid \psi_{n,\mathbf{k},\sigma}\rangle\right|^2
$$

This is the **"tot" column** for atom $a$ in the PROCAR file, summed over all angular momentum
channels. It is available when `LORBIT = 11` or `12` is set in INCAR.

## Participation Ratio (P-ratio)

The **P-ratio** introduced by Kumagai *et al.* (*Phys. Rev. B* **103**, 104102, 2021) measures
the fraction of total wavefunction weight on the **defect neighbourhood** atoms $\mathcal{N}$:

$$
\text{P-ratio}(n,\mathbf{k},\sigma) =
\frac{\displaystyle\sum_{a \in \mathcal{N}} p_a}{\displaystyle\sum_a p_a}
$$

| P-ratio | Physical meaning |
|---------|-----------------|
| $\approx 0$ | Delocalized — bulk host state |
| $> 0.2$ | Defect-character state (conventional threshold) |
| $\approx 1$ | Fully localized on defect neighbours — deep defect level |

## Inverse Participation Ratio (IPR)

The **site-based electronic IPR** is a code-agnostic localization metric that does not require
knowledge of the defect position:

$$
\text{IPR}(n,\mathbf{k},\sigma) =
\frac{\displaystyle\sum_a p_a^2}{\left(\displaystyle\sum_a p_a\right)^2}
$$

| IPR value | Physical meaning |
|-----------|-----------------|
| $1/N$ | Perfectly delocalized over $N$ atoms |
| $1$ | Fully localized on a single atom |

## Distinction from the phonon IPR

DefectPL uses the term "IPR" in two distinct contexts:

| Context | Module | Formula | Input |
|---------|--------|---------|-------|
| Phonon mode localization | `defectpl.utils.calc_IPR` | $\sum_a p_a^2 / (\sum_a p_a)^2$ where $p_a = \sum_i e_{k,a,i}^2$ | Phonon eigenvectors |
| Electronic state localization | `defectpl.participation_ratio` | $\sum_a p_a^2 / (\sum_a p_a)^2$ where $p_a$ = PROCAR "tot" | PROCAR projections |

Both share the same mathematical form but differ in input data and physical interpretation.

## Neighbour resolution strategy

DefectPL selects defect-neighbourhood atoms using the best available source:

1. **`defect_structure_info.json`** — reads `neighbor_atom_indices` (defectpl-generated or
   pydefect-compatible).
2. **Distance search on POSCAR/CONTCAR** — finds atoms within `cutoff_radius` Å.
3. **Empty list** — if no structural information is available; only IPR is meaningful.

## PROCAR parsing

VASP writes PROCAR when `LORBIT = 11` (or `12`). DefectPL includes a pure-Python parser that does
not require pymatgen:

```python
from defectpl.participation_ratio import read_procar

data = read_procar("PROCAR", use_pymatgen=False)
# data["spin_1"]["kpt_1"]["band_112"]["tot_per_atom"] → list of floats
```

## In DefectPL

```python
from defectpl.participation_ratio import ParticipationRatioCalculator

calc = ParticipationRatioCalculator(
    procar="PROCAR",
    defect_entry="defect_entry.json",
    defect_structure_info="defect_structure_info.json",
    cutoff_radius=3.5,
)
result = calc.run()
calc.to_json("participation_ratio.json")

# Inspect top-10 most localized states
for row in calc.top_localized(n=10, metric="p_ratio"):
    print(row["spin"], row["band"], row["energy"], row["p_ratio"])
```
