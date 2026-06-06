"""
Unit tests for the constants.py module within the defectpl package layout.
"""

import math
import pytest
from defectpl.constants import (
    AMU2KG,
    ANG2M,
    ELEMENTARY_CHARGE,
    EV2J,
    EV2MEV,
    HBAR_EVS,
    HBAR_JS,
    PLANCK_CONSTANT,
    THZ2EV,
)

# ==============================================================================
# TEST CASES
# ==============================================================================

def test_foundational_constants():
    """
    Validates exact CODATA values for primary foundational constants.
    """
    assert ELEMENTARY_CHARGE == 1.602176634e-19
    assert PLANCK_CONSTANT == 6.62607015e-34
    assert AMU2KG == 1.6605390666e-27
    assert ANG2M == 1e-10
    assert EV2MEV == 1e3


def test_derived_energy_conversions():
    """
    Verifies that the Joule-to-eV link remains exactly tied to the elementary charge.
    """
    assert EV2J == ELEMENTARY_CHARGE


def test_reduced_planck_constants():
    """
    Verifies mathematical and physical consistency of h-bar variants.
    """
    # Expected hbar in J·s = h / 2pi
    expected_hbar_js = PLANCK_CONSTANT / (2.0 * math.pi)
    pytest.approx(HBAR_JS, abs=1e-40) == expected_hbar_js

    # Expected hbar in eV·s = hbar_js / e
    expected_hbar_evs = expected_hbar_js / ELEMENTARY_CHARGE
    pytest.approx(HBAR_EVS, abs=1e-22) == expected_hbar_evs


def test_frequency_to_energy_conversion():
    """
    Validates that the THz to eV conversion factor matches standard physics.
    
    Formula: E (eV) = (h * 10^12 Hz) / e
    """
    expected_thz2ev = (PLANCK_CONSTANT * 1e12) / ELEMENTARY_CHARGE
    assert math.isclose(THZ2EV, expected_thz2ev, rel_tol=1e-9)


def test_type_integrity():
    """
    Ensures all exported parameters are floating point targets for clean matrix operations.
    """
    constants_list = [
        ELEMENTARY_CHARGE,
        PLANCK_CONSTANT,
        AMU2KG,
        ANG2M,
        EV2MEV,
        EV2J,
        HBAR_JS,
        HBAR_EVS,
        THZ2EV,
    ]
    for constant in constants_list:
        assert isinstance(constant, float)