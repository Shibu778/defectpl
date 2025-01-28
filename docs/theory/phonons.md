# Understanding Phonons

Phonons are collective motion of atoms in the crystal. At absolute zero temperature, atoms in the crystal show zero point motion, however atoms vibrate at higher temperature results in presence of phonons. There are different experimental techniques to measure phonons. Raman and Infrared spectroscopies are used to study phonon spectra near the origin of reciprocal space, however, inelastic neutron and X-ray scattering can sample phonon spectra at general point in Brillouin zone. Phonons can also be studied from first principles and useful for many purposes because of its high predictability. Thermal properties such as heat capacity, thermal expansion, and thermal conductivity are studied through phonon calculations. In studying the optical properties of point defect, phonon calculations provide necessary vibronic information to calculate the full PL spectra, which include the sharp ZPL line and the phonon sidebands.

There are two ways of doing first principles phonon calculations, i.e., finite-displacement supercell approach (FDM) and density functional perturbation theory (DFPT) approach.  In both the cases user friendly codes like VASP or Quantum Espresso to perform the first principles calculation to solve the Schrodinger equation. Further, Phonopy code can be used to calculate the force constants in supercell from the dataset of displacements and forces in FDM approach and interpolate the force constants in reciprocal space by Fourier transform in DFPT approach. 

## Crystal Structure and Potential

A crystal consists of many unit cell. The unit cell number is labelled as $l$ and the equilibrium position of an atom in the unitcell is labelled by $\kappa$. The equilibrium position of atoms are denoted as $\textbf{R}_{l\kappa}^0$. $N$ is the number of unitcell in a crystal and $n_a$ is the number of atoms in each unitcell. $V_c$ and $NV_c$ are the volume of unitcell and crystal, respectively. 

# References
1. [Togo, Atsushi. "First-principles phonon calculations with phonopy and phono3py." Journal of the Physical Society of Japan 92.1 (2023): 012001.](https://journals.jps.jp/doi/10.7566/JPSJ.92.012001)