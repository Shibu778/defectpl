# 🚀 DefectPL Benchmark & Pipeline Examples Summary

This document provides a comprehensive summary of the benchmark calculations for the NV center in diamond system. It aggregates important properties, outlines the underlying high-performance hardware orchestration, and defines the structural layout of the source datasets.

**Source Directory Location:** `defectpl/examples/NV_diamond`  
**Data Repository Baseline:** `defectpl/data/NV_diamond`

---

## 📋 Comprehensive Calculation Summary Matrix

The table below tracks the physical scalar characteristics, electron-phonon coupling constraints, and configuration coordinate displacements ($Q$ and $R$) extracted across all evaluated DFT functionals and fractional excitation profiles.

| Functional_Setup | Pipeline_Mode | Calculation Run Mode | Zero-Phonon Line (ZPL) Energy | Total Huang-Rhys (HR) Factor | Debye-Waller (DW) Factor | Total Number of Atoms (natoms) | ZPL Broadening Factor (gamma) | Gaussian Broadening (sigma) | Energy Mesh Resolution | Mass-Weighted Delta Q (delQ) | Structural Delta R (delR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **pbe_out** | abs_gs_force_mode | Force Mode | 1.750 eV | 3.87203 | 0.020816 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **pbe_out** | gs_zpl_disp_mode | Displacement Mode | 1.750 eV | 2.92711 | 0.053552 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | 0.642559 | 0.183851 Å |
| **pbe_out** | zpl_ems_force_mode | Force Mode | 1.750 eV | 2.73149 | 0.065122 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **frac_pbe_out** | abs_gs_force_mode | Force Mode | 1.729 eV | 3.18620 | 0.041329 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **frac_pbe_out** | gs_zpl_disp_mode | Displacement Mode | 1.729 eV | 2.57465 | 0.076181 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | 0.603373 | 0.172432 Å |
| **frac_pbe_out** | zpl_ems_force_mode | Force Mode | 1.729 eV | 2.41795 | 0.089104 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **hse06_out** | abs_gs_force_mode | Force Mode | 1.991 eV | 5.02907 | 0.006545 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **hse06_out** | gs_zpl_disp_mode | Displacement Mode | 1.991 eV | 3.23112 | 0.039513 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | 0.667635 | 0.190834 Å |
| **hse06_out** | zpl_ems_force_mode | Force Mode | 1.991 eV | 3.43931 | 0.032087 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **frac_hse06_out** | abs_gs_force_mode | Force Mode | 2.190 eV | 4.48589 | 0.011267 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |
| **frac_hse06_out** | gs_zpl_disp_mode | Displacement Mode | 2.190 eV | 3.20348 | 0.040621 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | 0.667367 | 0.190678 Å |
| **frac_hse06_out** | zpl_ems_force_mode | Force Mode | 2.190 eV | 3.22968 | 0.039570 | 215 | 2 meV | 0.006 eV | 1000 pts/eV | *nan* | *nan* |

> 💡 **Note on Parameters:** `delQ` represents the mass-weighted configuration coordinate displacement, while `delR` tracks the true aggregate structural root-mean-square displacement between ground and excited states. These are only computed during the geometric `gs_zpl_disp_mode` runs.

---

## 🛠️ Computational Setup

All underlying electronic structure calculations and force evaluations were performed using **VASP 6.6.0** on an enterprise NVIDIA GPU platform.

### Hardware Infrastructure
* **GPU Architecture:** 4 × NVIDIA Tesla V100-SXM2 (32 GB HBM2 per GPU)
* **Compute Capability:** `cc70` (Volta)
* **Driver Version:** 550.90.12 (CUDA 12.4 supported by system)

### Compilation & Build Details
* **Compiler Toolchain:** NVIDIA HPC SDK v23.5 (`mpif90`, `mpicc`, `nvc++`)
* **Optimization Flags:** `-fast`
* **GPU Offloading:** Enabled via OpenACC (`-DACC_OFFLOAD`) and native CUDA (`-DNVCUDA`)
* **Multi-GPU Communication:** NVIDIA Collective Communications Library (`-DUSENCCL`)
* **Linked Libraries:** * Linear Algebra: `BLAS`, `LAPACK`, `ScaLAPACK`
  * CUDA Libraries: `cuBLAS`, `cuSolver`, `cuFFT`, `NCCL`
  * FFT Library: FFTW 3.3.10

### Job Parallelization & Execution
The workloads were parallelized across all 4 onboard GPUs on a single compute node (mapping 1 MPI rank per physical GPU) using the following operational environment directives:
* **Total MPI Ranks:** 4 (`mpirun -n 4`)
* **OpenMP Threads:** 1 (`OMP_NUM_THREADS=1`)

---

## 📂 Pipeline Directory Blueprint

For more explicit details, configuration settings, extracted text variables, or localized vector graphics, you can dive directly into the output folders of each respective functional run. Each directory is structured symmetrically containing standalone `.svg` figures, flat data matrices, and JSON checkpoints:

```text
defectpl/examples/NV_diamond/
├── frac_hse06_out/                 # Check here for Fractional HSE06 detailed runs
│   ├── abs_gs_force_mode/          # Absorption lineshape properties & plots
│   ├── data/                       # Cached matrices
│   ├── gs_zpl_disp_mode/           # Displaced configuration coordinate results
│   └── zpl_ems_force_mode/         # Emission lineshape properties & plots
│
├── frac_pbe_out/                   # Check here for Fractional PBE detailed runs
│   ├── abs_gs_force_mode/
│   ├── data/
│   ├── gs_zpl_disp_mode/
│   └── zpl_ems_force_mode/
│
├── hse06_out/                      # Check here for Equilibrium HSE06 detailed runs
│   ├── abs_gs_force_mode/
│   ├── data/
│   ├── gs_zpl_disp_mode/
│   └── zpl_ems_force_mode/
│
├── pbe_out/                        # Check here for Equilibrium PBE detailed runs
│   ├── abs_gs_force_mode/
│   ├── data/
│   ├── gs_zpl_disp_mode/
│   └── zpl_ems_force_mode/
│
├── pl_example_summary.py           # Verification parsing routine script
├── pl_summary_table.csv            # Extracted row matrix in CSV format
├── pl_summary_table.md             # Rendered overview markdown data file
└── run_pl_examples.py              # Main execution wrapper tool for the engine
