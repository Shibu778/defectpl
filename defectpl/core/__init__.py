# -*- coding: utf-8 -*-
"""
defectpl.core — code-agnostic data containers and type definitions.

Sub-modules
-----------
structures    PhononData and EigenvalData dataclasses.
types         TypedDict definitions for structured dictionaries.
"""

from defectpl.core.structures import EigenvalData, PhononData

__all__ = [
    "EigenvalData",
    "PhononData",
]
