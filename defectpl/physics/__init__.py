# -*- coding: utf-8 -*-
"""
defectpl.physics — code-agnostic physics engines.

Sub-modules
-----------
lineshape            PL lineshape (Photoluminescence, VibrationalSpectra1D, CCD).
phonon               Phonon data helpers (read_band_yaml, force constants, etc.).
participation_ratio  Electronic participation ratio calculations.
"""

from defectpl.physics.lineshape import (
    ConfigurationCoordinateDiagram,
    Photoluminescence,
    VibrationalSpectra1D,
)
from defectpl.physics.phonon import read_band_yaml
from defectpl.physics.participation_ratio import (
    ParticipationRatioCalculator,
    compute_participation_ratios,
)

__all__ = [
    "ConfigurationCoordinateDiagram",
    "ParticipationRatioCalculator",
    "Photoluminescence",
    "VibrationalSpectra1D",
    "compute_participation_ratios",
    "read_band_yaml",
]
