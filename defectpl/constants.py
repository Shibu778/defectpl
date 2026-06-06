# -*- coding: utf-8 -*-
r"""
Physical constants and unit conversion factors for the defectpl package.

References and underlying constant values align precisely with the CODATA 
internationally recommended values. Derived parameters are calculated using 
exact conversion formulas to maintain absolute mathematical consistency across 
energy and mass spaces.

Attributes
----------
ELEMENTARY_CHARGE : float
    The elementary charge ($e$) in Coulombs (C).
PLANCK_CONSTANT : float
    The Planck constant ($h$) in Joule-seconds (J·s).
HBAR_JS : float
    The reduced Planck constant ($\hbar$) in Joule-seconds (J·s).
HBAR_EVS : float
    The reduced Planck constant ($\hbar$) in Electron-volt-seconds (eV·s).
THZ2EV : float
    Conversion factor transforming frequency in Terahertz (THz) directly into 
    energy equivalents in Electron-volts (eV). Derived as $h \times 10^{12} / e$.
AMU2KG : float
    Conversion factor mapping Unified Atomic Mass Units (amu or Da) into 
    Kilograms (kg).
ANG2M : float
    Conversion factor scaling length from Angstroms (Å) to Meters (m).
EV2MEV : float
    Conversion factor scaling Electron-volts (eV) to Millielectron-volts (meV).
EV2J : float
    Conversion factor mapping Electron-volts (eV) into Joules (J). Identical 
    to the numeric scale of the elementary charge.
"""

# =====================================================================
# Foundational CODATA Framework Definitions
# =====================================================================

ELEMENTARY_CHARGE: float = 1.602176634e-19
"""Elementary charge (e) in Coulombs (C). Exact CODATA definition."""

PLANCK_CONSTANT: float = 6.62607015e-34
"""Planck constant (h) in Joule-seconds (J·s). Exact CODATA definition."""

AMU2KG: float = 1.6605390666e-27
"""Conversion factor from Atomic Mass Units (amu) to Kilograms (kg)."""

ANG2M: float = 1e-10
"""Conversion factor from Angstroms (Å) to Meters (m). Exact metric scale."""

EV2MEV: float = 1e3
"""Conversion factor from Electron-volts (eV) to Millielectron-volts (meV)."""


# =====================================================================
# Derived Constants & Conversion Factors
# =====================================================================

# Exact relationship: 1 eV is defined by moving 1e through a 1V potential
EV2J: float = ELEMENTARY_CHARGE
"""Conversion factor from Electron-volts (eV) to Joules (J)."""

# Hbar in J·s: h / (2 * pi)
HBAR_JS: float = PLANCK_CONSTANT / (2.0 * 3.141592653589793)
"""Reduced Planck constant (hbar) in Joule-seconds (J·s)."""

# Hbar in eV·s: hbar_js / ev_to_j
HBAR_EVS: float = HBAR_JS / EV2J
"""Reduced Planck constant (hbar) in Electron-volt-seconds (eV·s)."""

# THz to eV: E = h * nu -> (Planck_constant * 1e12 Hz) / eV_conversion
THZ2EV: float = (PLANCK_CONSTANT * 1e12) / EV2J
"""Conversion factor from Terahertz (THz) to Electron-volts (eV)."""
