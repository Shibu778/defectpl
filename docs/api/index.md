# API Reference

All public symbols in DefectPL are documented here.
The pages use [mkdocstrings](https://mkdocstrings.github.io/) to render docstrings automatically —
hover over a symbol name in your IDE or click "Source" in the rendered page to see the
implementation.

| Module | Purpose |
|--------|---------|
| [`defectpl.defectpl`](photoluminescence.md) | `Photoluminescence`, `VibrationalSpectra1D`, `ConfigurationCoordinateDiagram` |
| [`defectpl.phonon`](phonon.md) | `GammaPhononData`, force-constant and band-yaml utilities |
| [`defectpl.utils`](utils.md) | Pure-math: $\Delta Q$, $S_k$, generating function, IPR |
| [`defectpl.participation_ratio`](participation_ratio.md) | P-ratio / IPR from PROCAR |
| [`defectpl.ks_analysis`](ks_analysis.md) | Kohn–Sham eigenvalue analysis and plotting |
| [`defectpl.plot`](plot.md) | `Plotter` — all visualization methods |
| [`defectpl.vasp`](vasp.md) | VASP file I/O (OUTCAR, EIGENVAL) |
| [`defectpl.defect_utils`](defect_utils.md) | `defect_entry.json`, `defect_structure_info.json` generators |
| [`defectpl.constants`](constants.md) | Physical constants (CODATA) |
