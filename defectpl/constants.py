# -*- coding: utf-8 -*-
"""
Physical constants and unit conversion factors for the defectpl package.
References align with CODATA recommended values.
"""

# =====================================================================
# Unit Conversion Factors
# =====================================================================

THZ2EV: float = 4.135667696e-3
"""Conversion factor from Terahertz (THz) to Electron-volts (eV)."""

AMU2KG: float = 1.6605390666e-27
"""Conversion factor from Atomic Mass Units (amu) to Kilograms (kg)."""

ANG2M: float = 1e-10
"""Conversion factor from Angstroms (Å) to Meters (m)."""

EV2MEV: float = 1e3
"""Conversion factor from Electron-volts (eV) to Millielectron-volts (meV)."""

EV2J: float = 1.602176634e-19
"""Conversion factor from Electron-volts (eV) to Joules (J)."""


# =====================================================================
# Fundamental Physical Constants
# =====================================================================

HBAR_JS: float = 1.054571817e-34
"""Reduced Planck constant (hbar) in Joule-seconds (J·s)."""

HBAR_EVS: float = 6.582119569e-16
"""Reduced Planck constant (hbar) in Electron-volt-seconds (eV·s)."""