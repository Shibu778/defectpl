# Photoluminescence

# General Photoluminescence Theory

Photoluminescence (PL) is a process in which materials emit light when excited by an external source of energy, such as light or heat. According to the general theory of PL, when an electron in a material ascends to a high-energy state and then returns to its ground state, a photon is emitted.

# The Underlying Physics and Approximations

The physical processes involved in PL can be explained using several approximations:

1. **Fermi's Golden Rule**: This principle dictates the probability of an electronic transition between two states, thus helping to estimate the intensity of the PL spectrum.

2. **Electric Dipole Approximation**: According to this approximation, the electric dipole moment of the material is behind the PL emission.

3. **Born-Oppenheimer Approximation**: This approximation separates the electron's degrees of freedom from its nuclear degrees, making it simpler to calculate the material's potential energy surfaces.

4. **Franck-Condon Approximation**: This approximation, which assumes that electronic transition happens faster than nuclear movements, aids in the calculation of both the Franck-Condon (FC) integrals and the PL spectrum.

# Weak Electron-Phonon Coupling in Photoluminescence

In cases of weak electron-phonon coupling, the PL spectrum is characterized by narrow shape with coinciding peaks for absorption and emission. This implies that nuclear movements have a negligible effect on the electronic transition. In such cases, the PL spectrum is ascertained using the following equation:

    G(ω) = C ω^3 ∑ w_m(T) |⟨χ_fm|χ_in⟩|^2 δ(E_ZPL + ℏω_fm - ℏω_in + ℏω)

# Strong Electron-Phonon Coupling in Photoluminescence

In conditions of strong electron-phonon coupling, the PL spectrum has a broad pattern with different peak positions for absorption and emission. This indicates that the electronic transition is significantly influenced by nuclear movements. In such cases, the PL spectrum is determined using the following equation:

    G(ω) = C ω^3 ∑ w_m(T) |⟨χ_fm|χ_in⟩|^2 δ(E_ZPL + ℏω_fm - ℏω_in + ℏω) e^(-S(ω))

# A One-Dimensional Configuration Coordinate Approach

The one-dimensional configuration coordinate approach, which includes the following steps, is used when dealing with strong electron-phonon coupling:

1. **Optimize the ground state and excited state geometries** - Quantum chemistry methods are used to derive optimal configurations for the ground state and excited state of the material.

2. **Build a 1D configuration coordinate diagram** - This diagram is created using linear interpolation between the optimized ground state and excited state geometries.

3. **Calculate the effective phonon modes and the Huang-Rhys factor** - Using the optimized geometries, these values are calculated.

4. **Compute the Franck-Condon integrals and the PL spectrum** - Using the effective phonon modes and the Huang-Rhys factor, these predictive details are calculated.

# Intermediate Electron-Phonon Coupling in Photoluminescence

When electron-phonon coupling is intermediate, the PL spectrum is broad, but not as broad as in the case of strong coupling. In this scenario, the following equation is used to determine the PL spectrum:

    G(ω) = C ω^3 ∑ w_m(T) |⟨χ_fm|χ_in⟩|^2 δ(E_ZPL + ℏω_fm - ℏω_in + ℏω) e^(-S(ω))

# Further Concepts and Methodologies

- **Multidimensional Nuclear Wavefunction**: This mathematical representation is used to portray nuclear motion in the presence of electron-phonon coupling.

- **Displaced Harmonic Oscillator Approximation**: This approximation, effective in situations with weak electron-phonon coupling, assumes that the nuclear motion can be represented as a displaced harmonic oscillator.

- **Generating Function**: This mathematical method aids in the computation of both FC integrals and the PL spectrum.

- **Spectral Density of Electron-Phonon (el-ph) Coupling**: This quantifies the distribution of phonon modes and their corresponding energies.

- **Partial Huang-Rhys Factor (HRF)**: This value measures the strength of electron-phonon coupling for a specific phonon mode.