# -*- coding: utf-8 -*-
"""
defectpl.physics.lineshape — PL lineshape and vibrational spectra.

Re-exports the core physics classes from :mod:`defectpl.defectpl` so they
are discoverable under the new ``physics`` namespace while keeping
backward-compatibility (patches in tests still target ``defectpl.defectpl.*``).
"""

from defectpl.defectpl import (
    ConfigurationCoordinateDiagram,
    Photoluminescence,
    VibrationalSpectra1D,
)

__all__ = [
    "ConfigurationCoordinateDiagram",
    "Photoluminescence",
    "VibrationalSpectra1D",
]
