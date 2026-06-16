# Configuration Coordinate Diagram

## The one-dimensional effective coordinate

In the **adiabatic approximation** the total energy of a defect system depends parametrically on
the nuclear positions $\{\mathbf{R}_a\}$. For an optical transition between the ground state (g)
and an excited state (e), the key coordinate is the **mass-weighted configuration coordinate
difference** [Alkauskas 2014b]:

$$
\Delta Q = \sqrt{\sum_a m_a |\mathbf{R}_{e,a} - \mathbf{R}_{g,a}|^2}
$$

Units: $\sqrt{\text{amu}}\cdot\text{Å}$.
$\Delta Q$ collapses the full $3N$-dimensional displacement into a single scalar that enters all
one-mode lineshape formulae.

The coordinate $\Delta R$ is the unweighted displacement norm:

$$
\Delta R = \sqrt{\sum_{a,i} (R_{e,a,i} - R_{g,a,i})^2}
$$

## Mode-projected coordinates

For each Gamma-point phonon mode $k$ the projection coordinate is [Alkauskas 2014a, Eq. 6]:

$$
q_k = \sum_{a,i} \sqrt{m_a}\,(R_{e,a,i} - R_{g,a,i})\,e_{k,a,i}
$$

and the corresponding **partial Huang–Rhys factor** is:

$$
S_k = \frac{\omega_k\, q_k^2}{2\hbar}
$$

The closure relation $\Delta Q^2 = \sum_k q_k^2$ holds when phonon eigenvectors form a complete
orthonormal set, providing a consistency check.

## Force-mode alternative

When only vertical forces (not relaxed structures) are available, $q_k$ can be obtained from the
force difference $\Delta\mathbf{F}_a = \mathbf{F}_{e,a} - \mathbf{F}_{g,a}$ at the ground-state
geometry [Alkauskas 2014a, Eq. 7]:

$$
q_k = \frac{1}{\omega_k^2} \sum_{a,i} \frac{\Delta F_{a,i}}{\sqrt{m_a}}\, e_{k,a,i}
$$

The two paths are equivalent under the harmonic approximation because
$\Delta\mathbf{F} = -\mathbf{H}\,(\mathbf{R}_e - \mathbf{R}_g)$ where $\mathbf{H}$ is the
mass-unweighted Hessian.

## Potential energy surfaces and CCD

The harmonic **Configuration Coordinate Diagram** plots:

$$
E_g(Q) = \tfrac{1}{2}\,\omega_g^2\,(Q - Q_g)^2
\qquad
E_e(Q) = E_\text{ZPL} + \tfrac{1}{2}\,\omega_e^2\,(Q - Q_e)^2
$$

Key energies extracted from this diagram:

| Symbol | Name | Definition |
|--------|------|-----------|
| $E_\text{ZPL}$ | Zero-phonon line energy | $E_e(Q_e) - E_g(Q_g)$ |
| $E_\text{rel}^{(g)}$ | Ground-state relaxation energy | $E_g(Q_e) - E_g(Q_g)$ |
| $E_\text{rel}^{(e)}$ | Excited-state relaxation energy | $E_e(Q_g) - E_e(Q_e)$ |
| $E_\text{abs}$ | Vertical absorption energy | $E_\text{ZPL} + E_\text{rel}^{(e)}$ |
| $E_\text{em}$ | Vertical emission energy | $E_\text{ZPL} - E_\text{rel}^{(g)}$ |
| Stokes shift | $E_\text{abs} - E_\text{em}$ | $E_\text{rel}^{(g)} + E_\text{rel}^{(e)}$ |

## Effective frequency

The effective phonon frequency $\omega_\text{eff}$ and partial HR factor $S_\text{eff}$ can be
obtained by fitting the ground-state parabola:

$$
\omega_\text{eff} = \sqrt{\frac{2\,E_\text{rel}^{(g)}}{\Delta Q^2}}
\qquad
S_\text{eff} = \frac{E_\text{rel}^{(g)}}{\hbar\omega_\text{eff}}
$$

For a multi-mode spectrum the **spectral-function-weighted** effective frequency is:

$$
\omega_\text{eff} = \frac{\sum_k S_k\,\omega_k}{\sum_k S_k} = \frac{\int \omega\,S(\omega)\,d\omega}{\int S(\omega)\,d\omega}
$$

## In DefectPL

```python
from defectpl.defectpl import ConfigurationCoordinateDiagram
from pymatgen.core import Structure

ccd = ConfigurationCoordinateDiagram(
    ground_struct=Structure.from_file("CONTCAR_gs"),
    excited_struct=Structure.from_file("CONTCAR_es"),
)

# Generate interpolated structures for DFT single-point calculations
ccd.generate_structures(n_steps=7, out_dir="ccd_structures/")

# After running VASP on each interpolated geometry:
ccd.analyze_ccd(
    gs_runs=["ccd_structures/gs_0/vasprun.xml", ...],
    es_runs=["ccd_structures/es_0/vasprun.xml", ...],
    de=1.945,
    save_plot="ccd.pdf",
)
```

CLI equivalent:

```bash
defectpl setup-ccd \
    --gs CONTCAR_GS --es CONTCAR_ES \
    --tmpl_gs template_gs/ --tmpl_es template_es/ \
    --steps "-0.2,0.0,0.4,0.6,0.8,1.0,1.2"

defectpl analyze-ccd \
    --gs CONTCAR_GS --es CONTCAR_ES \
    --gs_runs "run_0/vasprun.xml run_1/vasprun.xml" \
    --es_runs "run_0/vasprun.xml run_1/vasprun.xml" \
    --de 1.945 --save_plot ccd_fit.pdf
```
