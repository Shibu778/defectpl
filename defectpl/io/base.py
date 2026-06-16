# -*- coding: utf-8 -*-
"""
Reader Protocol definitions for code-agnostic DFT I/O.

Define :class:`PhononReader` and :class:`ElectronicReader` as
``runtime_checkable`` Protocol classes so any VASP/QE/ABINIT/CP2K reader
can be duck-typed without inheriting from a concrete base class.

New code readers (QE, ABINIT, CP2K, …) implement these Protocols and can
be dropped in wherever the physics layer expects a reader, without touching
any physics code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from defectpl.core.structures import EigenvalData, PhononData


@runtime_checkable
class PhononReader(Protocol):
    """Protocol for reading phonon data from any DFT code output."""

    def read_band_yaml(self, path: str) -> "PhononData":
        """Parse a band-structure file and return a code-agnostic PhononData."""
        ...

    def read_force_constants(self, path: str) -> "np.ndarray":
        """Read force constants from a FORCE_CONSTANTS-style file."""
        ...

    def read_forces(self, path: str, state: str = "gs") -> "np.ndarray":
        """Extract atomic forces from a run output (OUTCAR, xml, …)."""
        ...


@runtime_checkable
class ElectronicReader(Protocol):
    """Protocol for reading electronic structure data from any DFT code output."""

    def read_eigenvalues(self, path: str, k_idx: int = 0) -> "EigenvalData":
        """Parse eigenvalue/occupancy data at a selected k-point."""
        ...

    def read_structures(self, path: str) -> "tuple[list, list]":
        """Extract all ionic-step structures and forces from a trajectory file."""
        ...
