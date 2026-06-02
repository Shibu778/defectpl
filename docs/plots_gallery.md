# 📊 Visualizations Gallery

DefectPL provides an automated plotting suite backed by a custom, publication-optimized style profile (`defectpl.mplstyle`). The engine natively formats all graphics for single-column journal layouts using high-contrast, colorblind-safe color mapping and pristine LaTeX typography rendering.

Below are the key diagnostic plots you can generate using the package, organized by analytical tracks.

---

## 🌌 Track 1: Emission Intensities & Peak Profiles
These plots capture the macroscopic optical output of your quantum emitter, tracking the transition lineshape from multi-phonon couplings down to individual core configuration coordinate equivalents.

| **Intensity vs. Phonon Energy** | **One-Dimensional Vibrational Spectra** |
| :------------------------------: | :-------------------------------------: |
| ![intensity-photon-energy]       | ![oned]                                 |
| *Macroscopic photoluminescence intensity sideband.* | *Effective single-mode configuration coordinate representation.* |

---

## 🎛️ Track 2: Multi-Variable Matrix Analysis
These figures combine a twin $y$-axis framework with continuous color-mapped scatter overlays or unified curves. They are designed to instantly show the relationships between energy levels, coupling matrices ($S_k$), and localized defect dynamics.

| **Spectral Function, Partial HR Factor, & Localization Ratio** | **Spectral Function, Partial HR Factor, & IPR** |
| :----------------------------------------------------------: | :---------------------------------------------: |
| ![somega-pHR-locrat-penergy]                                 | ![S_ipr]                                        |
| *$S(\omega)$ and $S_k$ mapped by atomic spatial localization.* | *$S(\omega)$ and $S_k$ mapped by Inverse Participation Ratio.* |

| **Joint Spectral Function & Partial HR Sideband** |
| :--------------------------------------------------: |
| ![S_omega_Sks]                                      |
| *Clean dual-axis overlay matching continuous $S(\omega)$ directly against discrete $S_k$ weights.* |

---

## 🔬 Track 3: Lattice Displacements & Energies
These diagnostics inspect the physical lattice dynamics under the hood, showing you exactly where the crystal structure deforms upon vertical electronic transition.

| **Generalized Vibrational Displacement** | **Phonon Eigenvalues vs. Mode** |
| :---------------------------------------: | :-----------------------------: |
| ![vibrational-displacement]               | ![phonon-energy]                |
| *Structural shift projection $q_k$ across the spectrum.* | *Sequential energy distribution across all available modes.* |

---

## 🧩 Track 4: Phonon Localization Metrics
Isolated independent plots used to contrast localized deep-defect states against bulk-like host crystal vibrations.

| **Inverse Participation Ratio (IPR)** | **Localization Ratio** |
| :-----------------------------------: | :--------------------: |
| ![ipr]                                | ![loc_ratio]           |
| *Spikes pinpoint pristine, isolated defect-core modes.* | *Normalized metric tracking spatial energy confinement decay.* |

---

## 📈 Track 5: Standalone Coupling Fingerprints

| **Partial HR Factor ($S_k$) Only** | **Pure Spectral Density Function $S(\omega)$** |
| :---------------------------------: | :-------------------------------------------: |
| ![pHR]                              | ![S_pHR]                                      |
| *Raw coupling weight contribution per mode.* | *Continuous coupling function prior to peak convolution.* |

---

## 🔗 Image References

[intensity-photon-energy]: plots/intensity_vs_penergy.svg
[somega-pHR-locrat-penergy]: plots/S_omega_HRf_loc_rat_vs_penergy.svg
[vibrational-displacement]: plots/qk_vs_penergy.svg
[phonon-energy]: plots/penergy_vs_pmode.svg
[ipr]: plots/ipr_vs_penergy.svg
[loc_ratio]: plots/loc_rat_vs_penergy.svg
[pHR]: plots/HR_factor_vs_penergy.svg
[S_pHR]: plots/S_omega_vs_penergy.svg
[S_ipr]: plots/S_omega_HRf_ipr_vs_penergy.svg
[S_omega_Sks]: plots/S_omega_Sks_vs_penergy.svg
[oned]: plots/one_d_lineshape.svg