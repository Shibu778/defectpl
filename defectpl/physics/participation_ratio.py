# -*- coding: utf-8 -*-
"""
defectpl.physics.participation_ratio — electronic participation ratio.

Re-exports from :mod:`defectpl.participation_ratio` for the new namespace.
"""

from defectpl.participation_ratio import (
    ParticipationRatioCalculator,
    compute_participation_ratios,
    flatten_pr_result,
    neighbors_from_defect_structure_info,
    neighbors_from_structure,
    plot_pr_vs_band_index,
    plot_pr_vs_energy,
    read_procar,
    resolve_neighbors,
)

__all__ = [
    "ParticipationRatioCalculator",
    "compute_participation_ratios",
    "flatten_pr_result",
    "neighbors_from_defect_structure_info",
    "neighbors_from_structure",
    "plot_pr_vs_band_index",
    "plot_pr_vs_energy",
    "read_procar",
    "resolve_neighbors",
]
