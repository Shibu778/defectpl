# -*- coding: utf-8 -*-
"""
TypedDict definitions for structured dictionaries used across defectpl.

These types are used for type-checking only (``TYPE_CHECKING`` guard on
callers) and do not add any runtime overhead.
"""

from __future__ import annotations

from typing import List, TypedDict

import numpy as np


class PhononReaderResult(TypedDict):
    """Return type for raw phonon data before wrapping in PhononData."""

    frequencies: "np.ndarray"
    eigenvectors: "np.ndarray"
    masses: "np.ndarray"


class PLRunConfig(TypedDict, total=False):
    """Keyword configuration for a Photoluminescence run."""

    EZPL: float
    gamma: float
    resolution: int
    max_energy: float
    sigma: float
    T: float


class CCDConfig(TypedDict, total=False):
    """Configuration for a Configuration Coordinate Diagram analysis."""

    dE: float
    xlim: "tuple[float, float]"
    ylim: "tuple[float, float]"
    figsize: "tuple[float, float]"
    save_plot: str
