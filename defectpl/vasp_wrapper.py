# -*- coding: utf-8 -*-
"""
defectpl.vasp_wrapper — backward-compatible re-export shim.

All VASP workflow helpers now live in :mod:`defectpl.io.vasp`.
This module is kept for backward compatibility so that existing
``from defectpl.vasp_wrapper import ...`` calls continue to work.
"""

from defectpl.io.vasp import (
    VaspReader,
    analyze_ccd_framework,
    calc_dF,
    calc_dR,
    calc_delta_Q,
    generate_ccd_calculations,
    get_q_from_structure,
    prepare_dF_files,
    run_dynamic_yaml_comparison,
    run_kohn_sham_analysis,
    run_pl_calc_vasp_displacement_mode,
    run_pl_calc_vasp_force_mode,
)

__all__ = [
    "VaspReader",
    "analyze_ccd_framework",
    "calc_dF",
    "calc_dR",
    "calc_delta_Q",
    "generate_ccd_calculations",
    "get_q_from_structure",
    "prepare_dF_files",
    "run_dynamic_yaml_comparison",
    "run_kohn_sham_analysis",
    "run_pl_calc_vasp_displacement_mode",
    "run_pl_calc_vasp_force_mode",
]
