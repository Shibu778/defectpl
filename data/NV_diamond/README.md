## Computational Setup

All calculations were performed using **VASP 6.6.0** on an NVIDIA GPU platform.

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
The jobs were parallelized across all 4 GPUs on a single node (1 MPI rank per GPU) using the following configuration:
* **Total MPI Ranks:** 4 (`mpirun -n 4`)
* **OpenMP Threads:** 1 (`OMP_NUM_THREADS=1`)