# -*- coding: utf-8 -*-
"""
VASP and Phonopy data interface adapter routines for defectpl.
Ensures rigorous unit handling between pymatgen outputs, phonopy, and defectpl.

Author : Shibu Meher
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from defectpl.constants import THZ2EV

# Rigorous spectroscopic conversion constant: 1 cm^-1 = 1.2398419843e-4 eV (based on h*c)
CM1_TO_EV = 1.2398419843e-4


def extract_phonon_dfpt_phonopy(
    vasprun_path: Union[str, Path] = "vasprun.xml",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts Gamma-point phonon frequencies and eigenvectors from a VASP DFPT 
    calculation using ONLY a standalone vasprun.xml file via Phonopy diagonalization.

    Units Handling:
    ===============
    - Input Force Constants: Extracted by pymatgen in eV/Å^2.
    - Phonopy Frequencies: Diagonalized and returned as linear frequencies in THz.
    - Returned Frequencies: Scaled via THZ2EV into eV.
    - Returned Eigenvectors: Dimensionless normalized displacement vectors.
    """
    # Parse the standalone vasprun file natively using Pymatgen
    vrun = Vasprun(
        str(vasprun_path), 
        parse_dos=False, 
        parse_eigen=False, 
        parse_projected_eigen=False
    )
    
    if not hasattr(vrun, "force_constants") or vrun.force_constants is None:
        raise ValueError(
            f"The file '{vasprun_path}' does not contain an internal Hessian/dynmat block. "
            "Ensure your VASP calculation was executed with IBRION = 7 or 8."
        )

    structure = vrun.final_structure
    masses = np.array([site.specie.atomic_mass for site in structure], dtype=float)
    
    unitcell = PhonopyAtoms(
        symbols=[site.specie.symbol for site in structure],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
        masses=masses
    )

    # Initialize Phonopy with calculator="vasp" so it assumes standard VASP units (eV, Å, AMU)
    phonon = Phonopy(
        unitcell=unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3, dtype=float),
        calculator="vasp"
    )
    
    # Load force constants (eV/Å^2) into Phonopy
    phonon.force_constants = vrun.force_constants

    # Diagonalize the matrix at Gamma. Outputs freq_raw in THz.
    q_gamma = np.array([0.0, 0.0, 0.0], dtype=float)
    freq_raw, eig_raw = phonon.get_frequencies_with_eigenvectors(q_gamma)

    # Clean acoustic/imaginary noise floor and scale from THz -> eV
    frequencies = np.array(freq_raw, dtype=float)
    frequencies[frequencies < 0.0] = 0.0
    frequencies *= THZ2EV

    # Reshape column eigenvectors to standard shape: (n_modes, n_atoms, 3)
    n_modes = len(frequencies)
    n_atoms = len(structure)
    eigenvectors = np.reshape(eig_raw.real.T, (n_modes, n_atoms, 3))

    return frequencies, eigenvectors


def extract_phonon_dfpt_pmg(
    vasprun_path: Union[str, Path] = "vasprun.xml",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts Gamma-point phonon frequencies and eigenvectors directly from 
    Pymatgen's parsed VASP dynmat block, bypassing Phonopy steps entirely.

    Units Handling:
    ===============
    - Pymatgen Frequencies: Parsed natively into wavenumbers (cm^-1).
    - Returned Frequencies: Scaled via CM1_TO_EV directly into eV.
    - Returned Eigenvectors: Dimensionless Cartesian displacement fractions.
    """
    vrun = Vasprun(
        str(vasprun_path),
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False
    )

    if not hasattr(vrun, "normalmode_eigenvals") or vrun.normalmode_eigenvals is None:
        raise ValueError(
            f"The file '{vasprun_path}' does not contain computed normal mode items. "
            "Ensure the calculation was run with IBRION = 5, 6, 7, or 8."
        )

    # vrun.normalmode_eigenvals are stored by pymatgen in cm^-1 (wavenumbers)
    freq_raw = np.array(vrun.normalmode_eigenvals, dtype=float)
    
    # Filter out imaginary modes (Pymatgen parses imaginary modes as negative real numbers or complex)
    # If complex, np.real() strips imaginary tags; negative bounds catch remaining instabilities
    frequencies = np.where(freq_raw > 0.0, freq_raw, 0.0)
    
    # Convert directly from cm^-1 to eV
    frequencies *= CM1_TO_EV

    # Extract eigenvectors; Pymatgen already shaped these to (n_modes, n_atoms, 3)
    eigenvectors = np.array(vrun.normalmode_eigenvecs, dtype=float)

    return frequencies, eigenvectors

if __name__ == "__main__":
    vasprun_file = "../../../../dfpt_phonon/vasprun.xml"  # Adjust path as needed
    freqs_phonopy, eigs_phonopy = extract_phonon_dfpt_phonopy(vasprun_file)
    print("Frequencies (eV):", freqs_phonopy)
    print("Eigenvectors shape:", eigs_phonopy.shape)