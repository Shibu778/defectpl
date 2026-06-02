# -*- coding: utf-8 -*-
"""
Unit tests for the phonon.py module within the defectpl package layout.
"""

from pathlib import Path
import numpy as np
import pytest

# ==============================================================================
# CONFIGURATION / INPUT VARIABLES
# ==============================================================================
BASE_TEST_DIR = Path(__file__).parent
TEST_DIR = BASE_TEST_DIR / "temp_test_phonon_data"
MOCK_BAND_YAML_PATH = TEST_DIR / "band.yaml"

# Import targets securely after path normalization
from defectpl.phonon import (
    GammaPhononData,
    read_band_yaml,
    extract_gamma_phonon_data,
)
from defectpl.constants import THZ2EV

# ==============================================================================
# FIXTURES & MOCK DATA GENERATION
# ==============================================================================

@pytest.fixture(scope="module", autouse=True)
def setup_mock_phonon_files():
    """
    Builds a structured, syntax-accurate Phonopy band.yaml text stream prior to testing.

    Yields
    ------
    None
        Control is handed over to the executing test suite components.
    """
    # Safeguard folder setup
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Generate an artificial, valid Phonopy band.yaml layout (2 atoms, 6 modes)
    band_yaml_content = (
        "natom: 2\n"
        "nqpoint: 1\n"
        "npath: 1\n"
        "points:\n"
        "- symbol: Ga\n"
        "  mass: 69.723\n"
        "- symbol: As\n"
        "  mass: 74.922\n"
        "phonon:\n"
        "- q-position: [ 0.0000000,  0.0000000,  0.0000000 ]\n"
        "  distance:  0.0000000\n"
        "  band:\n"
        "  - # mode 1 (Acoustic noise, should bound to 0.0)\n"
        "    frequency: -0.0500000000\n"
        "    eigenvector:\n"
        "    - [ [  0.707106,  0.0 ], [  0.0,  0.0 ], [  0.0,  0.0 ] ]\n"
        "    - [ [  0.707106,  0.0 ], [  0.0,  0.0 ], [  0.0,  0.0 ] ]\n"
        "  - # mode 2\n"
        "    frequency: 3.5000000000\n"
        "    eigenvector:\n"
        "    - [ [  0.0,  0.0 ], [  0.707106,  0.0 ], [  0.0,  0.0 ] ]\n"
        "    - [ [  0.0,  0.0 ], [  0.707106,  0.0 ], [  0.0,  0.0 ] ]\n"
        "  - # mode 3\n"
        "    frequency: 3.5000000000\n"
        "    eigenvector:\n"
        "    - [ [  0.0,  0.0 ], [  0.0,  0.0 ], [  0.707106,  0.0 ] ]\n"
        "    - [ [  0.0,  0.0 ], [  0.0,  0.0 ], [  0.707106,  0.0 ] ]\n"
        "  - # mode 4\n"
        "    frequency: 8.2000000000\n"
        "    eigenvector:\n"
        "    - [ [  0.5,  0.0 ], [  0.0,  0.0 ], [  0.0,  0.0 ] ]\n"
        "    - [ [ -0.5,  0.0 ], [  0.0,  0.0 ], [  0.0,  0.0 ] ]\n"
        "  - # mode 5\n"
        "    frequency: 8.2000000000\n"
        "    eigenvector:\n"
        "    - [ [  0.0,  0.0 ], [  0.5,  0.0 ], [  0.0,  0.0 ] ]\n"
        "    - [ [  0.0,  0.0 ], [ -0.5,  0.0 ], [  0.0,  0.0 ] ]\n"
        "  - # mode 6\n"
        "    frequency: 8.8000000000\n"
        "    eigenvector:\n"
        "    - [ [  0.0,  0.0 ], [  0.0,  0.0 ], [  0.5,  0.0 ] ]\n"
        "    - [ [  0.0,  0.0 ], [  0.0,  0.0 ], [ -0.5,  0.0 ] ]\n"
    )

    with open(MOCK_BAND_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(band_yaml_content)

    yield

    # Comprehensive structural cleanup operations
    if MOCK_BAND_YAML_PATH.is_file():
        MOCK_BAND_YAML_PATH.unlink()
    if TEST_DIR.is_dir():
        TEST_DIR.rmdir()


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_read_band_yaml():
    """
    Validates that frequencies are converted to eV, negative acoustic modes are 
    clamped to 0.0, and eigenvectors are correctly unphased and flattened.
    """
    freqs, evecs, masses = read_band_yaml(MOCK_BAND_YAML_PATH, q_idx=0)

    # 1. Frequency validation
    assert len(freqs) == 6
    assert freqs[0] == 0.0  # Imaginary/acoustic boundary check (-0.05 -> 0.0)
    assert freqs[1] == pytest.approx(3.5 * THZ2EV)
    assert freqs[5] == pytest.approx(8.8 * THZ2EV)

    # 2. Flattened Matrix check: shape must be (nmodes, natoms * 3) -> (6, 2 * 3) = (6, 6)
    assert evecs.shape == (6, 6)
    
    # Mode 4 specific layout validation: [0.5, 0, 0, -0.5, 0, 0]
    np.testing.assert_allclose(evecs[3], [0.5, 0.0, 0.0, -0.5, 0.0, 0.0], atol=1e-5)

    # 3. Masses check
    assert len(masses) == 2
    np.testing.assert_allclose(masses, [69.723, 74.922])


def test_extract_gamma_phonon_data():
    """
    Ensures the factory function successfully generates an instance of 
    GammaPhononData with full list conversions.
    """
    phonon_data = extract_gamma_phonon_data(MOCK_BAND_YAML_PATH)

    assert isinstance(phonon_data, GammaPhononData)
    assert phonon_data.natoms == 2
    assert phonon_data.nmodes == 6
    assert len(phonon_data.frequencies) == 6
    assert isinstance(phonon_data.frequencies, list)
    assert isinstance(phonon_data.eigenvectors, list)
    
    # Check that nested vectors are flat (length 6)
    assert len(phonon_data.eigenvectors[0]) == 6
    assert phonon_data.meta_info["source_file"] == str(MOCK_BAND_YAML_PATH)


def test_gamma_phonon_data_serialization():
    """
    Validates that the GammaPhononData container matches MSONable specifications
    and completely survives round-trip dictionary serialization.
    """
    phonon_data = extract_gamma_phonon_data(MOCK_BAND_YAML_PATH)
    
    # Run serialization path
    data_dict = phonon_data.as_dict()
    assert data_dict["@module"] == "defectpl.phonon"
    assert data_dict["@class"] == "GammaPhononData"
    assert data_dict["natoms"] == 2
    assert data_dict["nmodes"] == 6

    # Run deserialization path
    reconstructed_data = GammaPhononData.from_dict(data_dict)
    assert isinstance(reconstructed_data, GammaPhononData)
    assert reconstructed_data.natoms == phonon_data.natoms
    assert reconstructed_data.nmodes == phonon_data.nmodes
    np.testing.assert_allclose(reconstructed_data.frequencies, phonon_data.frequencies)
    np.testing.assert_allclose(reconstructed_data.eigenvectors, phonon_data.eigenvectors)
    np.testing.assert_allclose(reconstructed_data.masses, phonon_data.masses)