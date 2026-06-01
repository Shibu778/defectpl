# -*- coding: utf-8 -*-
"""
Unit tests for mathematical and physical algorithm functions inside utils.py.
"""

from pathlib import Path
import numpy as np
import pytest
from pymatgen.core import Structure

from defectpl.utils import (
    calc_delR,
    calc_delQ,
    calc_qks,
    calc_Sks,
    gaussian_broadening,
    calc_S_omega,
    calc_IPR,
    calc_St,
    calc_Gts,
    calc_Spectrum_Intensity,
    calc_delta_Q,
    calculate_hermite,
    calculate_overlap_element,
)

# =====================================================================
# PATH CONFIGURATION SETUP FOR VASP INPUT FILES
# =====================================================================
DATA_DIR = Path(__file__).parent / "data"
POSCAR_GS_PATH = DATA_DIR / "pbe" / "gs" / "CONTCAR"
POSCAR_ES_PATH = DATA_DIR / "pbe" / "zpl" / "CONTCAR"


# =====================================================================
# MATHEMETHICAL ALGORITHM FIXTURES AND TESTS
# =====================================================================

@pytest.fixture
def mock_structural_data():
    """Generates simple arrays tracking mock displacements, masses, and modes."""
    masses = np.array([12.011, 15.999])  # Carbon and Oxygen in AMU
    # 2 atoms, 3 dimensions
    dR = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.2, -0.1]
    ])
    # 1 normal mode eigenvector slice with shape (nmodes, natoms, 3)
    eigenvectors = np.array([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]])
    frequencies = np.array([0.05])  # Mode frequency in eV
    return masses, dR, eigenvectors, frequencies


def test_calc_delR():
    dR = np.array([[1.0, 2.0, -2.0], [0.0, 0.0, 0.0]])
    # root-sum-squared: sqrt(1^2 + 2^2 + (-2)^2) = sqrt(9) = 3.0
    assert calc_delR(dR) == pytest.approx(3.0)


def test_calc_delQ(mock_structural_data):
    masses, dR, _, _ = mock_structural_data
    # Atom 1 dR^2 sum = 0.01. Atom 2 dR^2 sum = 0.04 + 0.01 = 0.05
    # Expected: sqrt(12.011 * 0.01 + 15.999 * 0.05)
    expected = np.sqrt(12.011 * 0.01 + 15.999 * 0.05)
    assert calc_delQ(masses, dR) == pytest.approx(expected)


def test_calc_qks(mock_structural_data):
    masses, dR, eigenvectors, _ = mock_structural_data
    qks = calc_qks(masses, dR, eigenvectors)
    assert isinstance(qks, np.ndarray)
    assert len(qks) == 1
    assert qks[0] > 0.0


def test_calc_Sks():
    qks = np.array([0.5])
    frequencies = np.array([0.1])
    Sks = calc_Sks(qks, frequencies)
    assert isinstance(Sks, np.ndarray)
    assert Sks[0] > 0.0


def test_gaussian_broadening():
    # Test single evaluation point center alignment peak
    val = gaussian_broadening(omega=2.0, omega_k=2.0, sigma=0.1)
    expected_peak = 1.0 / (0.1 * np.sqrt(2.0 * np.pi))
    assert val == pytest.approx(expected_peak)

    # Test array broadcast behavior
    omegas = np.array([1.9, 2.0, 2.1])
    vals = gaussian_broadening(omegas, 2.0, 0.1)
    assert len(vals) == 3
    assert vals[1] > vals[0]


def test_calc_S_omega():
    frequencies = np.array([0.1, 0.2])
    Sks = np.array([1.5, 0.5])
    omega_range = [0.0, 0.5, 100]  # start, stop, points
    
    S_omega = calc_S_omega(frequencies, Sks, omega_range, sigma=6e-3)
    assert isinstance(S_omega, np.ndarray)
    assert len(S_omega) == 100
    assert np.any(S_omega > 0.0)


def test_calc_IPR():
    # Fully localized single vector entry mode check
    eigenvectors = np.array([[
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]])
    ipr = calc_IPR(eigenvectors)
    assert ipr[0] == pytest.approx(1.0)


def test_time_domain_transforms():
    S_omega = np.ones(64)  # Simple flat frequency input vector array block
    St = calc_St(S_omega)
    assert len(St) == 64
    assert np.iscomplexobj(St)

    # Validate downstream generation function calculations step paths
    Gts = calc_Gts(Sts=St, total_HR=2.5, gamma=2.0, resolution=100.0)
    assert len(Gts) == 64
    
    # Test back transformation spectrum profiles intensities maps
    A, A_intensity = calc_Spectrum_Intensity(Gts, EZPL=1.95, resolution=100.0)
    assert len(A) == 64
    assert len(A_intensity) == 64


# =====================================================================
# PYMATGEN STRUCTURE DEPENDENT VASP FILE PARSER TESTS
# =====================================================================

def test_calc_delta_Q_mismatched_structures():
    # Enforce failure rule limits check bounds
    s1 = Structure([[10, 0, 0], [0, 10, 0], [0, 0, 10]], ["C"], [[0, 0, 0]])
    s2 = Structure([[10, 0, 0], [0, 10, 0], [0, 0, 10]], ["C", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    with pytest.raises(ValueError, match="Structures must have the same number of atoms."):
        calc_delta_Q(s1, s2)


def test_calc_delta_Q_with_vasp_data():
    """Validates physical delta Q evaluation using structure coordinates in tests/data."""
    if not (POSCAR_GS_PATH.exists() and POSCAR_ES_PATH.exists()):
        pytest.skip("VASP structural sample tracking testing files missing from your data folder path.")

    struct_gs = Structure.from_file(POSCAR_GS_PATH)
    struct_es = Structure.from_file(POSCAR_ES_PATH)

    delta_q = calc_delta_Q(struct_gs, struct_es)
    
    assert isinstance(delta_q, float)
    assert delta_q >= 0.0  # Displacement norm must be non-negative


# =====================================================================
# ANALYTICAL HERMITE AND FRANCK-CONDON OVERLAP INTEGRAL TESTS
# =====================================================================

def test_calculate_hermite():
    # Known exact mathematical evaluations for Physicist Hermite Polynomial entries
    assert calculate_hermite(0, 5.3) == pytest.approx(1.0)
    assert calculate_hermite(1, 1.5) == pytest.approx(3.0)  # 2 * x = 2 * 1.5 = 3
    assert calculate_hermite(2, 2.0) == pytest.approx(14.0)  # 4 * x^2 - 2 = 16 - 2 = 14


def test_calculate_overlap_element():
    # Simple evaluation baseline check testing identical wells bounds transitions setup
    m, n = 0, 0
    rho = 0.0
    cosfi, sinfi = 1.0, 0.0  # Orthogonal system overlap boundary settings limit configurations
    
    ix = calculate_overlap_element(m, n, rho, cosfi, sinfi)
    assert isinstance(ix, float)