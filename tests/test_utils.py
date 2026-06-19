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
    calc_phonon_occupation,
    calc_C_omega,
    calc_Ct,
    calc_C_total,
    calc_Absorption_Intensity,
    calc_effective_phonon_frequency,
    calc_IPR_alkauskas,
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
    dR = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, -0.1]])
    # 1 normal mode eigenvector slice with shape (nmodes, natoms, 3)
    eigenvectors = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
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
    # Fully localized on atom 0: IPR = 1
    eigenvectors = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    ipr = calc_IPR(eigenvectors)
    assert ipr[0] == pytest.approx(1.0)

    # Fully delocalized over 2 atoms (equal weight): IPR = 1/N = 0.5
    # Each atom gets weight 0.5 in a normalized eigenvector.
    w = np.sqrt(0.5)
    eigenvectors_deloc = np.array([[[w, 0.0, 0.0], [w, 0.0, 0.0]]])
    ipr_deloc = calc_IPR(eigenvectors_deloc)
    assert ipr_deloc[0] == pytest.approx(0.5)

    # Un-normalized eigenvectors: formula must not depend on overall scale.
    # Scale the localized eigenvector by 3 — IPR must still be 1.
    eigenvectors_scaled = np.array([[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    ipr_scaled = calc_IPR(eigenvectors_scaled)
    assert ipr_scaled[0] == pytest.approx(1.0)


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
    s2 = Structure(
        [[10, 0, 0], [0, 10, 0], [0, 0, 10]], ["C", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    with pytest.raises(
        ValueError, match="Structures must have the same number of atoms."
    ):
        calc_delta_Q(s1, s2)


def test_calc_delta_Q_with_vasp_data():
    """Validates physical delta Q evaluation using structure coordinates in tests/data."""
    if not (POSCAR_GS_PATH.exists() and POSCAR_ES_PATH.exists()):
        pytest.skip(
            "VASP structural sample tracking testing files missing from your data folder path."
        )

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
    cosfi, sinfi = (
        1.0,
        0.0,
    )  # Orthogonal system overlap boundary settings limit configurations

    ix = calculate_overlap_element(m, n, rho, cosfi, sinfi)
    assert isinstance(ix, float)


# =====================================================================
# TEMPERATURE-DEPENDENT FUNCTIONS
# =====================================================================


def test_calc_phonon_occupation_zero_temperature():
    freqs = np.array([0.02, 0.04, 0.06])
    nks = calc_phonon_occupation(freqs, temperature=0.0)
    np.testing.assert_array_equal(nks, np.zeros(3))


def test_calc_phonon_occupation_high_temperature():
    # At very high T, n̄_k ≈ k_B T / ħω → large positive values
    freqs = np.array([0.01, 0.02])
    nks = calc_phonon_occupation(freqs, temperature=1e6)
    assert np.all(nks > 0)
    # Higher T → lower frequency → larger occupation
    assert nks[0] > nks[1]


def test_calc_phonon_occupation_bose_einstein_limit():
    # At room temperature for a mid-range phonon, check non-zero occupation
    freqs = np.array([0.05])  # 50 meV phonon
    nks = calc_phonon_occupation(freqs, temperature=300.0)
    assert nks[0] > 0.0
    # Reference: x = 0.05 / (8.617e-5 * 300) ≈ 1.934 → n̄ ≈ 0.169
    assert nks[0] == pytest.approx(0.169, abs=0.01)


def test_calc_C_omega_zero_temperature():
    freqs = np.array([0.02, 0.04])
    Sks = np.array([0.5, 0.3])
    nks = np.zeros(2)
    omega_range = [0.0, 0.08, 500]
    C = calc_C_omega(freqs, Sks, nks, omega_range, sigma=6e-3)
    assert C.shape == (500,)
    np.testing.assert_array_equal(C, np.zeros(500))


def test_calc_C_omega_finite_temperature():
    freqs = np.array([0.02, 0.04])
    Sks = np.array([0.5, 0.3])
    nks = np.array([1.0, 0.5])
    omega_range = [0.0, 0.08, 500]
    C = calc_C_omega(freqs, Sks, nks, omega_range, sigma=6e-3)
    assert C.shape == (500,)
    assert np.all(C >= 0.0)
    assert np.max(C) > 0.0


def test_calc_Ct_zero_temperature():
    C_omega = np.zeros(1000)
    Ct = calc_Ct(C_omega)
    assert Ct.shape == (1000,)
    np.testing.assert_allclose(Ct, np.zeros(1000), atol=1e-10)


def test_calc_Ct_is_real():
    freqs = np.array([0.02, 0.04])
    Sks = np.array([0.5, 0.3])
    nks = np.array([0.5, 0.2])
    omega_range = [0.0, 0.08, 500]
    C_omega = calc_C_omega(freqs, Sks, nks, omega_range, sigma=6e-3)
    Ct = calc_Ct(C_omega)
    assert np.isrealobj(Ct) or np.all(np.imag(Ct) == 0)


def test_calc_C_total_zero_occupation():
    nks = np.zeros(3)
    Sks = np.array([1.0, 0.5, 0.2])
    result = calc_C_total(nks, Sks)
    assert result == pytest.approx(0.0)


def test_calc_C_total_dot_product():
    nks = np.array([1.0, 2.0, 3.0])
    Sks = np.array([0.1, 0.2, 0.3])
    result = calc_C_total(nks, Sks)
    assert result == pytest.approx(0.1 + 0.4 + 0.9)


def test_calc_Gts_temperature_zero_matches_default():
    # With Cts=None (old API) and temperature=0 (Cts=zeros, C_total=0),
    # the result should be identical
    n = 200
    Sts = np.linspace(-1, 1, n)
    total_HR = 1.0
    gamma = 0.01
    resolution = 100

    Gt_old = calc_Gts(Sts, total_HR, gamma, resolution)
    Ct_zero = np.zeros(n)
    Gt_new = calc_Gts(Sts, total_HR, gamma, resolution, Cts=Ct_zero, C_total=0.0)
    np.testing.assert_allclose(Gt_old, Gt_new, rtol=1e-12)


def test_calc_Absorption_Intensity_shape():
    n = 500
    Gts = np.ones(n, dtype=complex)
    A_abs, intensity_abs = calc_Absorption_Intensity(Gts, EZPL=2.0, resolution=100)
    assert A_abs.shape == (n,)
    assert intensity_abs.shape == (n,)


def test_calc_Absorption_Intensity_omega1_prefactor():
    n = 500
    Gts = np.ones(n, dtype=complex)
    A_abs, intensity_abs = calc_Absorption_Intensity(Gts, EZPL=1.0, resolution=100)
    # intensity = A_abs * ω¹; check ratio where A_abs is non-negligible
    j = np.arange(n)
    omega = j / 100.0
    np.testing.assert_allclose(np.abs(intensity_abs), np.abs(A_abs) * omega, rtol=1e-10)


def test_calc_effective_phonon_frequency_uniform():
    freqs = np.array([0.02, 0.04, 0.06])
    dq = np.array([1.0, 1.0, 1.0])
    omega_eff = calc_effective_phonon_frequency(freqs, dq)
    expected = float(np.sum(freqs**2 * dq**2) / np.sum(dq**2))
    assert omega_eff == pytest.approx(expected)


def test_calc_effective_phonon_frequency_zero_raises():
    freqs = np.array([0.02, 0.04])
    dq = np.zeros(2)
    with pytest.raises(ValueError, match="zero"):
        calc_effective_phonon_frequency(freqs, dq)


def test_calc_IPR_alkauskas_reciprocal():
    # IPR_alkauskas = 1 / IPR_trad when both are computed for the same eigenvectors
    eigenvectors = np.random.rand(5, 3, 3)
    ipr_alk = calc_IPR_alkauskas(eigenvectors)
    ipr_trad = calc_IPR(eigenvectors)
    np.testing.assert_allclose(ipr_alk, 1.0 / ipr_trad, rtol=1e-10)


def test_calc_IPR_alkauskas_range():
    # Alkauskas IPR lies in [1, N] where N = natoms
    natoms = 4
    eigenvectors = np.abs(np.random.rand(6, natoms, 3)) + 0.01
    ipr_alk = calc_IPR_alkauskas(eigenvectors)
    assert np.all(ipr_alk >= 1.0 - 1e-10)
    assert np.all(ipr_alk <= natoms + 1e-10)
