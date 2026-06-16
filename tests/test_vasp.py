# -*- coding: utf-8 -*-
"""
Unit tests for the vasp.py module within the defectpl package layout.
"""

from pathlib import Path
import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Eigenval

# ==============================================================================
# CONFIGURATION / INPUT VARIABLES
# ==============================================================================
# Graphically isolate and anchor test paths relative to this file's directory
BASE_TEST_DIR = Path(__file__).parent
TEST_DIR = BASE_TEST_DIR / "temp_test_vasp_data"
MOCK_OUTCAR_PATH = TEST_DIR / "OUTCAR"
MOCK_POSCAR_PATH = TEST_DIR / "POSCAR"
MOCK_EIGENVAL_PATH = TEST_DIR / "EIGENVAL"

# Import targets securely after path normalization
from defectpl.vasp import (
    OutcarParser,
    check_outcar_convergence,
    get_final_structure_and_forces_from_outcar,
    get_first_structure_and_forces_from_outcar,
    get_nions,
    get_species_and_index_map,
    get_structures_and_forces,
    get_spin_multiplicity,
    read_eigenval_file,
)

# ==============================================================================
# FIXTURES & MOCK DATA GENERATION
# ==============================================================================


@pytest.fixture(scope="module", autouse=True)
def setup_mock_vasp_files():
    """
    Builds structured, synthetic VASP text streams prior to testing.

    Yields
    ------
    None
        Control is handed over to the executing test suite components.
    """
    # Safeguard folder setup
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate an artificial, syntax-accurate OUTCAR file stream
    outcar_content = (
        " POTCAR:  PAW_PBE Ga 05Jan2006                   \n"
        " POTCAR:  PAW_PBE As 05Jan2006                   \n"
        " NIONS =       2\n"
        " ions per type =               1   1\n"
        " VOLUME and BASIS-vectors are now :\n"
        " -----------------------------------------------------------------------------\n"
        "  energy-cutoff  :      400.00\n"
        "  volume of cell :       45.00\n"
        "      direct lattice vectors                 reciprocal lattice vectors\n"
        "     5.000000  0.000000  0.000000         0.200000  0.000000  0.000000\n"
        "     0.000000  5.000000  0.000000         0.000000  0.200000  0.000000\n"
        "     0.000000  0.000000  5.000000         0.000000  0.000000  0.200000\n"
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------\n"
        "      0.00000   0.00000   0.00000         0.01000   0.02000   0.03000\n"
        "      1.25000   1.25000   1.25000        -0.01000  -0.02000  -0.03000\n"
        " -----------------------------------------------------------------------------\n"
        " VOLUME and BASIS-vectors are now :\n"
        " -----------------------------------------------------------------------------\n"
        "  energy-cutoff  :      400.00\n"
        "  volume of cell :       45.00\n"
        "      direct lattice vectors                 reciprocal lattice vectors\n"
        "     5.100000  0.000000  0.000000         0.196078  0.000000  0.000000\n"
        "     0.000000  5.100000  0.000000         0.000000  0.196078  0.000000\n"
        "     0.000000  0.000000  5.100000         0.000000  0.000000  0.196078\n"
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------\n"
        "      0.00000   0.00000   0.00000         0.00100   0.00200   0.00300\n"
        "      1.26000   1.26000   1.26000        -0.00100  -0.00200  -0.00300\n"
        " -----------------------------------------------------------------------------\n"
        " reached required accuracy - stopping structural energy minimization\n"
        " Total CPU time used (sec):     120.50\n"
    )
    with open(MOCK_OUTCAR_PATH, "w", encoding="utf-8") as f:
        f.write(outcar_content)

    # 2. Generate a baseline matching POSCAR file
    poscar_content = (
        "GaAs Mock Structure\n"
        "1.0\n"
        "5.0 0.0 0.0\n"
        "0.0 5.0 0.0\n"
        "0.0 0.0 5.0\n"
        "Ga As\n"
        "1 1\n"
        "Cartesian\n"
        "0.00 0.00 0.00\n"
        "1.25 1.25 1.25\n"
    )
    with open(MOCK_POSCAR_PATH, "w", encoding="utf-8") as f:
        f.write(poscar_content)

    # 3. Generate a baseline matching spin-polarized EIGENVAL file
    eigenval_content = (
        "     215     215     500       2\n"
        "  0.5669542E+01  0.1068224E-08  0.1068223E-08  0.1068223E-08  0.5000000E-15\n"
        "  1.000000000000000E-004\n"
        "  CAR\n"
        "unknown system\n"
        "    862      1     10\n"  # Configured here with 10 bands for testing purposes
        "  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.1000000E+01\n"
        "    1      -14.543596    -14.537239   1.000000   1.000000\n"
        "    2      -13.646644    -13.643089   1.000000   1.000000\n"
        "    3      -13.444986    -13.442122   1.000000   1.000000\n"
        "    4      -13.444986    -13.442121   1.000000   1.000000\n"
        "    5      -13.399910    -13.391155   1.000000   1.000000\n"
        "    6      -13.399908    -13.391154   1.000000   1.000000\n"
        "    7      -13.308602    -13.293761   1.000000   1.000000\n"
        "    8      -12.731101    -12.727239   1.000000   1.000000\n"  # This will be the HOMO (idx 7)
        "    9        1.500000      1.200000   0.000000   0.000000\n"  # Added Empty Band: LUMO (idx 8)
        "   10        5.500000      5.100000   0.000000   0.000000\n"
    )
    with open(MOCK_EIGENVAL_PATH, "w", encoding="utf-8") as f:
        f.write(eigenval_content)

    yield

    # Comprehensive structural cleanup operations
    for mock_file in [MOCK_OUTCAR_PATH, MOCK_POSCAR_PATH, MOCK_EIGENVAL_PATH]:
        if mock_file.is_file():
            mock_file.unlink()
    if TEST_DIR.is_dir():
        TEST_DIR.rmdir()


# ==============================================================================
# OUTCAR & TRAJECTORY TEST CASES
# ==============================================================================


def test_check_outcar_convergence():
    """Validates whether convergence flags are correctly parsed from the file tail."""
    status = check_outcar_convergence(MOCK_OUTCAR_PATH)
    assert isinstance(status, dict)
    assert status["structural_converged"] is True
    assert status["electronic_converged"] is True
    assert status["finished_cleanly"] is True


def test_check_outcar_convergence_filenotfound():
    """Ensures an explicit FileNotFoundError is raised if a file path is wrong."""
    with pytest.raises(FileNotFoundError):
        check_outcar_convergence(TEST_DIR / "NONEXISTENT_OUTCAR")


def test_get_nions():
    """Tests extraction of the atom count token."""
    nions = get_nions(MOCK_OUTCAR_PATH)
    assert nions == 2


def test_get_species_and_index_map():
    """Validates that POTCAR mappings successfully map back to sequential arrays."""
    species_map = get_species_and_index_map(MOCK_OUTCAR_PATH)
    assert species_map == ["Ga", "As"]


def test_get_structures_and_forces_native():
    """Tests trajectory tracking without external POSCAR help."""
    structures, forces = get_structures_and_forces(MOCK_OUTCAR_PATH)

    assert len(structures) == 2
    assert len(forces) == 2

    assert structures[0].lattice.abc == (5.0, 5.0, 5.0)
    assert structures[1].lattice.abc == (5.1, 5.1, 5.1)

    assert forces[0].shape == (2, 3)
    np.testing.assert_allclose(forces[0][0], [0.01, 0.02, 0.03])


def test_get_structures_and_forces_with_poscar():
    """Validates structure processing when explicit POSCAR overrides are provided."""
    structures, forces = get_structures_and_forces(
        MOCK_OUTCAR_PATH, poscar_path=MOCK_POSCAR_PATH
    )
    assert len(structures) == 2
    assert structures[0].species[0].symbol == "Ga"


def test_get_first_and_last_standalone():
    """Verifies index-targeted tracking wrapper shortcuts."""
    struct_first, force_first = get_first_structure_and_forces_from_outcar(
        MOCK_OUTCAR_PATH
    )
    struct_last, force_last = get_final_structure_and_forces_from_outcar(
        MOCK_OUTCAR_PATH
    )

    assert struct_first.lattice.abc == (5.0, 5.0, 5.0)
    assert struct_last.lattice.abc == (5.1, 5.1, 5.1)

    np.testing.assert_allclose(force_first[0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(force_last[0], [0.001, 0.002, 0.003])


def test_outcar_parser_class():
    """Tests structural features of the OutcarParser class."""
    parser = OutcarParser(MOCK_OUTCAR_PATH)
    assert parser.natoms == 2

    struct, forces = parser.get_final_structure_and_forces()
    assert isinstance(struct, Structure)
    assert forces.shape == (2, 3)

    conv = parser.check_convergence()
    assert conv["structural_converged"] is True


# ==============================================================================
# KOHN-SHAM & ELECTRONIC TEST CASES
# ==============================================================================


def test_get_spin_multiplicity():
    """Validates formula behavior ($2S + 1$) calculation logic."""
    # S = |3 - 1| / 2 = 1.0 -> 2(1.0) + 1 = 3.0
    assert get_spin_multiplicity(3, 1) == 3.0
    # S = |2 - 2| / 2 = 0.0 -> 2(0.0) + 1 = 1.0
    assert get_spin_multiplicity(2, 2) == 1.0


def test_read_eigenval_file(monkeypatch):
    """Verifies parsing of synthetic spin-polarized eigenvalue records."""
    # Mock eigenvalue band properties to match mock structural lines
    mock_band_props = [
        [14.23, 13.92],  # Gaps [up, down]
        [1.500, 1.200],  # LUMOs [up, down]
        [-12.73, -12.72],  # HOMOs [up, down]
    ]
    monkeypatch.setattr(Eigenval, "eigenvalue_band_properties", mock_band_props)

    data = read_eigenval_file(MOCK_EIGENVAL_PATH, k_idx=0)

    assert "up" in data
    assert "down" in data
    assert len(data["up"]) == 10  # Fixed: Updated from 4 to 10

    # Asserting indices based on 0-indexing
    assert data["homo_up_idx"] == 7  # Band 8
    assert data["lumo_up_idx"] == 8  # Band 9
    assert data["homo_down_idx"] == 7  # Band 8
    assert data["lumo_down_idx"] == 8  # Band 9

    # |7 - 7| / 2 = 0 -> Spin Multiplicity = 1.0
    assert data["spin_multiplicity"] == 1.0
    assert data["nelect"] == 862
