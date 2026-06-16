# -*- coding: utf-8 -*-
"""
Unit tests for the plot.py module within the defectpl package layout.
"""

from pathlib import Path
import numpy as np
import pytest

# Import targets securely after path normalization
from defectpl.plot import Plotter

# ==============================================================================
# CONFIGURATION / INPUT VARIABLES
# ==============================================================================
BASE_TEST_DIR = Path(__file__).parent
TEST_DIR = BASE_TEST_DIR / "temp_test_plot_data"

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_dir():
    """Ensures a clean temporary directory environment exists for plot artifacts."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Comprehensive cleanup of exported files
    for f in TEST_DIR.glob("*.*"):
        try:
            f.unlink()
        except OSError:
            pass
    if TEST_DIR.is_dir():
        TEST_DIR.rmdir()


@pytest.fixture
def dummy_phonon_data():
    """Generates synthetic phononic arrays matching 12 active modes."""
    np.random.seed(42)
    frequencies = np.linspace(0.001, 0.080, 12)  # 0 to 80 meV roughly in eV
    iprs = np.random.uniform(0.1, 0.9, 12)
    localization_ratio = np.random.uniform(0.0, 1.0, 12)
    qks = np.random.uniform(0.01, 0.5, 12)
    Sks = np.random.exponential(scale=0.5, size=12)

    # Continuous density profiles (omega values)
    omega_range = [0.0, 0.100, 100]  # Start, End, Steps
    S_omega = np.sin(np.linspace(0, np.pi, 100)) ** 2 * 500.0  # Dummy 1/eV curve

    return {
        "frequencies": frequencies,
        "iprs": iprs,
        "localization_ratio": localization_ratio,
        "qks": qks,
        "Sks": Sks,
        "omega_range": omega_range,
        "S_omega": S_omega,
    }


# ==============================================================================
# TEST CASES
# ==============================================================================


def test_plot_penergy_vs_pmode(dummy_phonon_data):
    """Verifies generation and exportation of phonon energy vs mode index plots."""
    plotter = Plotter()
    file_name = "test_penergy_vs_pmode"

    plotter.plot_penergy_vs_pmode(
        frequencies=dummy_phonon_data["frequencies"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="png",
    )

    expected_file = TEST_DIR / f"{file_name}.png"
    assert expected_file.is_file()


def test_plot_ipr_vs_penergy(dummy_phonon_data):
    """Verifies generation and exportation of Inverse Participation Ratio scatter plots."""
    plotter = Plotter()
    file_name = "test_ipr_vs_penergy"

    plotter.plot_ipr_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        iprs=dummy_phonon_data["iprs"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="pdf",
    )

    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()


def test_plot_loc_rat_vs_penergy(dummy_phonon_data):
    """Verifies generation and exportation of Mode Localization Ratio scatter plots."""
    plotter = Plotter()
    file_name = "test_loc_rat_vs_penergy"

    plotter.plot_loc_rat_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        localization_ratio=dummy_phonon_data["localization_ratio"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="png",
    )

    expected_file = TEST_DIR / f"{file_name}.png"
    assert expected_file.is_file()


def test_plot_qk_vs_penergy(dummy_phonon_data):
    """Verifies generation and exportation of configurational vibrational mode displacements."""
    plotter = Plotter()
    file_name = "test_qk_vs_penergy"

    plotter.plot_qk_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        qks=dummy_phonon_data["qks"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="pdf",
    )

    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()


def test_plot_hr_factor_vs_penergy(dummy_phonon_data):
    """Verifies generation and exportation of partial mode Huang-Rhys factor scatter profiles."""
    plotter = Plotter()
    file_name = "test_hr_factor_vs_penergy"

    plotter.plot_HR_factor_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        Sks=dummy_phonon_data["Sks"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="png",
    )

    expected_file = TEST_DIR / f"{file_name}.png"
    assert expected_file.is_file()


def test_plot_s_omega_vs_penergy(dummy_phonon_data):
    """Verifies generation and exportation of continuous spectral density graphs."""
    plotter = Plotter()
    file_name = "test_s_omega_vs_penergy"

    plotter.plot_S_omega_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        S_omega=dummy_phonon_data["S_omega"],
        omega_range=dummy_phonon_data["omega_range"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="pdf",
    )

    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()


def test_plot_s_omega_sks_vs_penergy(dummy_phonon_data):
    """Verifies generation of dual-axis overlaid spectral density curves and scatter plots."""
    plotter = Plotter()
    file_name = "test_s_omega_sks_vs_penergy"

    plotter.plot_S_omega_Sks_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        S_omega=dummy_phonon_data["S_omega"],
        omega_range=dummy_phonon_data["omega_range"],
        Sks=dummy_phonon_data["Sks"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="png",
    )

    expected_file = TEST_DIR / f"{file_name}.png"
    assert expected_file.is_file()


def test_plot_s_omega_sks_loc_rat_vs_penergy(dummy_phonon_data):
    """Verifies color-mapped localization profiles on secondary axes with colorbars."""
    plotter = Plotter()
    file_name = "test_s_omega_loc_rat"

    plotter.plot_S_omega_Sks_Loc_rat_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        S_omega=dummy_phonon_data["S_omega"],
        omega_range=dummy_phonon_data["omega_range"],
        Sks=dummy_phonon_data["Sks"],
        localization_ratio=dummy_phonon_data["localization_ratio"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        pylim=[0, None],
        fig_format="pdf",
    )

    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()


def test_plot_s_omega_sks_ipr_vs_penergy(dummy_phonon_data):
    """Verifies color-mapped IPR profiles on secondary axes with colorbars."""
    plotter = Plotter()
    file_name = "test_s_omega_ipr"

    plotter.plot_S_omega_Sks_ipr_vs_penergy(
        frequencies=dummy_phonon_data["frequencies"],
        S_omega=dummy_phonon_data["S_omega"],
        omega_range=dummy_phonon_data["omega_range"],
        Sks=dummy_phonon_data["Sks"],
        iprs=dummy_phonon_data["iprs"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="png",
    )

    expected_file = TEST_DIR / f"{file_name}.png"
    assert expected_file.is_file()


def test_plot_intensity_vs_penergy():
    """Verifies normalization scaling and clipping dimensions for clean lines."""
    plotter = Plotter()
    file_name = "test_intensity_vs_penergy"
    dummy_intensity = np.array([0.0, 0.1, 0.5, 1.0, 0.8, 0.3, 0.05, 0.0])

    plotter.plot_intensity_vs_penergy(
        frequencies=np.array([0.01, 0.02]),
        I=dummy_intensity,
        resolution=10,
        xlim=(0.0, 0.8),
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="pdf",
    )

    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()


def test_graceful_invalid_format_fallback(dummy_phonon_data):
    """Ensures fallback configuration executes normally when an invalid format string is passed."""
    plotter = Plotter()
    file_name = "test_fallback"

    plotter.plot_penergy_vs_pmode(
        frequencies=dummy_phonon_data["frequencies"],
        plot=False,
        out_dir=TEST_DIR,
        file_name=file_name,
        fig_format="invalid_extension_xyz",
    )

    # Method should fallback to saving as a .pdf layout gracefully
    expected_file = TEST_DIR / f"{file_name}.pdf"
    assert expected_file.is_file()
