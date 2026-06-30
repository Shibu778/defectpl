# TDM & WAVECAR Analysis Tutorial

This tutorial shows how to compute Transition Dipole Moments (TDMs),
Inverse Participation Ratios (IPRs), and optical properties (ZPL,
radiative lifetime) for point defects using `defectpl`'s WAVECAR tools.

---

## 1. Background

The **transition dipole moment** between two Kohn-Sham states $i$ and $j$
at k-point **k** is computed via the **p–r relation**:

$$
\langle\psi_j|\hat{r}|\psi_i\rangle
= \frac{-i}{\Delta E / (2\,\text{Ry})}
  \langle\psi_j|\hat{p}|\psi_i\rangle \cdot a_0 \cdot D_{\rm conv}
$$

where $\Delta E = E_j - E_i$, $a_0 = 0.529$\,Å (Bohr radius), and
$D_{\rm conv} = 2.5417$\,D/a.u. The momentum matrix element is

$$
\langle\psi_j|\hat{p}|\psi_i\rangle_k
= \sum_{\mathbf{G}} C_j^*(\mathbf{G})\,(\mathbf{k}+\mathbf{G})_{\rm cart}\,C_i(\mathbf{G})
$$

The Brillouin-zone (BZ) averaged TDM uses IBZKPT multiplicities.

The **Einstein A coefficient** (radiative emission rate, MHz) is

$$
A = \frac{n_r E_{\rm ZPL}^3 |\mu|^2}{3\pi\varepsilon_0 c^3\hbar^4}
$$

and the **radiative lifetime** is $\tau = 1/A$.

---

## 2. Quick-start

```python
from defectpl import WavecarReader, read_ibzkpt_weights

# Load WAVECAR (compressed .gz/.bz2 also supported)
wfc = WavecarReader("WAVECAR")
print(f"nspin={wfc.nspin}  nkpts={wfc.nkpts}  nbands={wfc.nbands}")

# Read k-point weights
kw = read_ibzkpt_weights("IBZKPT")

# TDM between bands 638 and 639 (1-based), BZ average
avg = wfc.get_weighted_avg_tdm(ispin=1, iband_i=638, iband_j=639, kweights=kw)
print(f"|TDM| (BZ avg) = {avg['avg_tdm_magnitude']:.4f} Debye")
print(f"ΔE   (BZ avg) = {avg['avg_dE']:.4f} eV")
```

---

## 3. Exploring bands near the Fermi level

Use `select_bands` to choose bands automatically:

```python
from defectpl import WavecarReader, select_bands, read_ibzkpt_weights

wfc = WavecarReader("WAVECAR")
kw  = read_ibzkpt_weights("IBZKPT")

# 10 occupied + 10 unoccupied bands closest to the Fermi level
bands = select_bands(wfc, ispin=1, mode="near_fermi", n_occ=10, n_unocc=10)
print("Selected band indices:", bands)

# Energy window mode (eV, absolute)
bands_ew = select_bands(wfc, ispin=1, mode="energy_window",
                         energy_min=3.0, energy_max=7.0)

# Explicit range
bands_rng = select_bands(wfc, ispin=1, mode="band_range", band_range=(630, 650))

# All occupied → unoccupied transitions in the selected pool
result = wfc.get_all_transitions(
    ispin=1, kweights=kw,
    mode="occupation",
    min_tdm=0.01,          # discard sub-0.01 D transitions
)

print(f"Unique band pairs: {result['metadata']['n_unique_pairs']}")
for entry in result["strongest_transitions"][:5]:
    print(f"  {entry['iband_i']:4d} → {entry['iband_j']:4d}  "
          f"|TDM|={entry['avg_tdm_magnitude']:.3f} D  "
          f"ΔE={entry['avg_dE']:.3f} eV")
```

---

## 4. Cross-state TDM (ΔSCF)

For defects where the ground and excited state require separate VASP runs
(ΔSCF approach), load two WAVECARs:

```python
from defectpl import WavecarReader, read_ibzkpt_weights
import numpy as np

wfc_gs = WavecarReader("WAVECAR_gs")
wfc_es = WavecarReader("WAVECAR_es")
kw     = read_ibzkpt_weights("IBZKPT")

# Cross-state TDM at all k-points
res = wfc_gs.get_tdm_cross_state_all_kpoints(wfc_es, ispin=1,
                                              iband_i=638, iband_j=639)

# BZ-weighted average
w   = kw / kw.sum()
avg_mag = float(np.dot(w, res["tdm_magnitude"]))
print(f"Cross-state |TDM| (BZ avg) = {avg_mag:.4f} Debye")
```

---

## 5. Inverse Participation Ratio (IPR)

IPR measures wavefunction localization:

$$
\text{IPR} = \frac{\sum_n |\phi(n)|^4}{\left(\sum_n |\phi(n)|^2\right)^2}
$$

```python
from defectpl import (WavecarReader, read_ibzkpt_weights,
                       compute_ipr_all, save_ipr_json, save_ipr_csv)

wfc = WavecarReader("WAVECAR")
kw  = read_ibzkpt_weights("IBZKPT")

# Compute IPR for 10+10 near-Fermi bands
result = compute_ipr_all(wfc, ispin=1, kweights=kw,
                          select_mode="near_fermi", n_occ=10, n_unocc=10)

# Save results
save_ipr_json(result, "ipr_result.json")
save_ipr_csv(result, "ipr_summary.csv")

# Print most-localised bands
top5 = sorted(result["band_summary"],
              key=lambda r: r["weighted_avg_ipr"], reverse=True)[:5]
for r in top5:
    print(f"Band {r['iband']:4d}  IPR={r['weighted_avg_ipr']:.3e}  "
          f"E={r['avg_energy']:.3f} eV")
```

---

## 6. Optical properties (ZPL, Einstein A, lifetime)

```python
from defectpl import compute_optical_properties

props = compute_optical_properties(
    g_path="/data/ground/",         # VASP ground-state directory
    e_path="/data/excited/",        # VASP excited-state directory
    tdm_gg=1.5,                     # same-state BZ-avg |TDM| (Debye)
    tdm_ge=1.3,                     # cross-state BZ-avg |TDM| (Debye)
    nr=2.42,                        # refractive index (diamond: 2.42)
)

print(f"ZPL       = {props['ZPL']:.4f} eV")
print(f"dQ        = {props['dQ']:.4f} amu^0.5·Å")
print(f"A (GG)    = {props['A_gg']:.3f} MHz")
print(f"τ (GG)    = {props['lifetime_gg']:.3f} ns")
print(f"A (GE)    = {props['A_ge']:.3f} MHz")
print(f"τ (GE)    = {props['lifetime_ge']:.3f} ns")
```

### Individual helpers

```python
from defectpl import get_zpl, get_dQ, get_einstein_coefficient, get_radiative_lifetime

zpl = get_zpl("/data/gs/", "/data/es/")
dQ  = get_dQ("/data/gs/",  "/data/es/")
A   = get_einstein_coefficient(zpl_ev=zpl, tdm_debye=1.5, nr=2.65)
tau = get_radiative_lifetime(A)
print(f"ZPL={zpl:.4f} eV   dQ={dQ:.4f} amu^0.5·Å   A={A:.3f} MHz   τ={tau:.3f} ns")
```

---

## 7. WAVECAR trimming for storage efficiency

Large WAVECARs can be trimmed to keep only selected bands:

```python
from defectpl import WavecarReader, select_bands, read_ibzkpt_weights

wfc   = WavecarReader("WAVECAR")
kw    = read_ibzkpt_weights("IBZKPT")
bands = select_bands(wfc, 1, mode="near_fermi", n_occ=10, n_unocc=10)

# Standard trim — keeps full nbands count but zeros non-selected coefficients
wfc.trim_save_wavecar(bands, outfile="WAVECAR_trim")

# Compact format — stores ONLY selected bands, auto-detected on read
wfc.save_compact_wavecar(bands, outfile="WAVECAR_compact")

# The compact WAVECAR is read exactly like a full one
wfc2 = WavecarReader("WAVECAR_compact")
res  = wfc2.get_tdm_all_kpoints(1, 638, 639)   # original band indices work
```

---

## 8. Visualisation

```python
from defectpl import WavecarReader, read_ibzkpt_weights
from defectpl.physics.tdm_viz import (
    plot_tdm_dashboard,
    plot_tdm_absorption,
    plot_ipr_scatter,
    save_wfc_vasp,
    save_wfc_vesta,
)
from defectpl import compute_ipr_all
from defectpl.io.wavecar import get_structure

wfc     = WavecarReader("WAVECAR")
kw      = read_ibzkpt_weights("IBZKPT")
tdm_res = wfc.get_tdm_all_kpoints(1, 638, 639)

# 2×3 summary dashboard
plot_tdm_dashboard(tdm_res, sigma=0.05, outfile="tdm_dashboard.pdf")

# Gaussian-broadened absorption spectrum
fig, E, spec = plot_tdm_absorption(tdm_res, sigma=0.05, outfile="absorption.png")

# IPR scatter plot
ipr_res = compute_ipr_all(wfc, 1, kweights=kw,
                           select_mode="near_fermi", n_occ=10, n_unocc=10)
plot_ipr_scatter(ipr_res, fermi_level=5.2, outfile="ipr_scatter.png")

# Export wavefunction for VESTA visualisation
struct = get_structure(".")
save_wfc_vasp(wfc, 1, 1, 638, struct, outfile="band638.vasp")
save_wfc_vesta(wfc, 1, 1, 638, struct, outfile="band638.vesta")
```

---

## 9. Compressed input files

All readers transparently handle `.gz`, `.bz2`, `.xz`, and `.lzma` files:

```python
from defectpl import WavecarReader, read_ibzkpt_weights
from defectpl.io.wavecar import read_oszicar, read_poscar

wfc = WavecarReader("WAVECAR.gz")       # decompressed to temp file
kw  = read_ibzkpt_weights("IBZKPT.gz")
E   = read_oszicar("OSZICAR.bz2")["final_energy"]
s   = read_poscar("POSCAR.xz")
```

---

## 10. Command-line interface

All major functions are exposed as `defectpl` sub-commands:

```bash
# TDM between two specific bands
defectpl tdm calc --iband-i 638 --iband-j 639 --ibzkpt IBZKPT --out tdm_638_639.json

# All occupied→unoccupied transitions
defectpl tdm all --mode occupation --top 10 --out all_tdm.json

# Cross-state TDM
defectpl tdm cross --wavecar-gs WAVECAR_gs --wavecar-es WAVECAR_es \
    --iband-i 638 --iband-j 639 --out cross_tdm.json

# Compact WAVECAR
defectpl tdm trim --bands 635-650 --compact --out WAVECAR_compact

# TDM plots
defectpl tdm plot --tdm-json tdm_638_639.json --plot-type dashboard --out dash.pdf
defectpl tdm plot --tdm-json tdm_638_639.json --plot-type absorption --sigma 0.03

# IPR calculation
defectpl ipr calc --mode near_fermi --n-occ 10 --n-unocc 10

# IPR plots
defectpl ipr plot --ipr-json ipr_result.json --plot-type scatter
defectpl ipr plot --ipr-json ipr_result.json --plot-type bar --top-n 15

# ZPL + radiative lifetime
defectpl zpl calc --ground /data/gs/ --excited /data/es/ \
    --tdm-gg 1.5 --tdm-ge 1.3 --nr 2.65 --out optical.json

# Real-space wavefunction export
defectpl wfc save --iband 638 --vesta
```

---

## 11. Complete workflow script

```python
"""
Complete TDM / IPR / optical-properties workflow for a NV-like defect.
"""
from pathlib import Path
import numpy as np
from defectpl import (
    WavecarReader, select_bands, read_ibzkpt_weights,
    compute_ipr_all, save_ipr_json, save_ipr_csv,
    compute_optical_properties,
)
from defectpl.physics.tdm_viz import (
    plot_tdm_dashboard, plot_ipr_scatter, save_wfc_vasp,
)
from defectpl.io.wavecar import get_structure

GROUND_DIR  = Path("/data/nv/ground/")
EXCITED_DIR = Path("/data/nv/excited/")
N_REFR      = 2.42   # diamond

# ── Read wavefunctions ──────────────────────────────────────────────────────
wfc_gs = WavecarReader(GROUND_DIR / "WAVECAR")
wfc_es = WavecarReader(EXCITED_DIR / "WAVECAR")
kw     = read_ibzkpt_weights(GROUND_DIR / "IBZKPT")
ef     = 5.20   # Fermi level from OUTCAR

# ── Band selection ──────────────────────────────────────────────────────────
bands = select_bands(wfc_gs, ispin=1, mode="homo_lumo_range",
                     below_homo_ev=2.0, above_lumo_ev=2.0,
                     fermi_level=ef)
print(f"Selected {len(bands)} bands: {bands}")

# ── Same-state TDM (GS→GS) ─────────────────────────────────────────────────
HOMO, LUMO = bands[len(bands)//2 - 1], bands[len(bands)//2]
avg_gg = wfc_gs.get_weighted_avg_tdm(1, HOMO, LUMO, kw)
tdm_gg = avg_gg["avg_tdm_magnitude"]
print(f"Same-state |TDM| = {tdm_gg:.4f} D  ΔE = {avg_gg['avg_dE']:.4f} eV")

# ── Cross-state TDM (GS→ES) ─────────────────────────────────────────────────
cross_res = wfc_gs.get_tdm_cross_state_all_kpoints(wfc_es, 1, HOMO, LUMO)
w       = kw / kw.sum()
tdm_ge  = float(np.dot(w, cross_res["tdm_magnitude"]))
print(f"Cross-state |TDM| = {tdm_ge:.4f} D")

# ── Optical properties ───────────────────────────────────────────────────────
props = compute_optical_properties(
    GROUND_DIR, EXCITED_DIR,
    tdm_gg=tdm_gg, tdm_ge=tdm_ge, nr=N_REFR,
)
print(f"ZPL       = {props['ZPL']:.4f} eV")
print(f"dQ        = {props['dQ']:.4f} amu^0.5·Å")
print(f"τ (GG)    = {props['lifetime_gg']:.2f} ns")
print(f"τ (GE)    = {props['lifetime_ge']:.2f} ns")

# ── IPR ─────────────────────────────────────────────────────────────────────
ipr_result = compute_ipr_all(wfc_gs, ispin=1, kweights=kw, bands=bands)
save_ipr_json(ipr_result, "ipr.json")
save_ipr_csv(ipr_result,  "ipr.csv")

# ── Plots ────────────────────────────────────────────────────────────────────
tdm_res = wfc_gs.get_tdm_all_kpoints(1, HOMO, LUMO)
plot_tdm_dashboard(tdm_res, sigma=0.04, outfile="tdm_dashboard.pdf")
plot_ipr_scatter(ipr_result, fermi_level=ef, outfile="ipr_scatter.pdf")

struct = get_structure(GROUND_DIR)
save_wfc_vasp(wfc_gs, 1, 1, HOMO, struct, outfile=f"band{HOMO}_density.vasp")
```

---

## 12. API Reference

| Symbol | Description |
|---|---|
| `WavecarReader` | Reads VASP WAVECAR binary |
| `VaspwaveH5Reader` | Reads vaspwave.h5 (VASP 6) |
| `select_bands` | Automatic band selection |
| `compute_ipr_band` | IPR of one KS state |
| `compute_ipr_all` | IPR of multiple states across all k |
| `compute_ipr_weighted` | k-weighted IPR for one band |
| `save_ipr_json` | Save IPR result to JSON |
| `save_ipr_csv` | Save IPR summary to CSV |
| `get_zpl` | Zero-phonon line from directories |
| `get_dQ` | Mass-weighted configuration shift |
| `get_einstein_coefficient` | Einstein A (MHz) |
| `get_radiative_lifetime` | Lifetime (ns) from A |
| `compute_optical_properties` | Full optical workflow |
| `plot_tdm_dashboard` | 2×3 TDM summary figure |
| `plot_tdm_absorption` | Gaussian-broadened spectrum |
| `plot_ipr_scatter` | IPR vs eigenvalue scatter |
| `save_wfc_vasp` | CHGCAR wavefunction export |
| `save_wfc_vesta` | VESTA project file |

All symbols are importable directly from `defectpl`:

```python
from defectpl import WavecarReader, compute_optical_properties, ...
```

or from the sub-package:

```python
from defectpl.physics.tdm import WavecarReader
from defectpl.io.wavecar import read_ibzkpt_weights
```
