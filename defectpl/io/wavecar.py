# -*- coding: utf-8 -*-
"""
defectpl.io.wavecar
===================
Transparent I/O helpers for VASP output files needed by the TDM / WAVECAR
analysis pipeline.

Supported compression: ``.gz``, ``.bz2``, ``.xz`` / ``.lzma``

Text files (IBZKPT, OSZICAR, OUTCAR, POSCAR/CONTCAR, KPOINTS):
    ``open_text(path)`` returns a text file-like object regardless of compression.

Binary files (WAVECAR):
    ``open_wavecar(path)`` decompresses to a tempfile and returns a seekable
    binary handle.  Caller must delete the temp file when done (or use as
    a context manager via :class:`WavecarHandle`).

VASP readers
------------
* ``read_ibzkpt_weights``  — k-point weights from IBZKPT
* ``read_oszicar``         — total energy from OSZICAR
* ``read_outcar_energy``   — ``energy(sigma->0)`` from OUTCAR
* ``read_outcar_fermi``    — Fermi level from OUTCAR
* ``read_poscar``          — minimal POSCAR/CONTCAR parser
* ``get_total_energy``     — auto-detect OSZICAR / OUTCAR / vasprun.xml
* ``get_fermi_level``      — auto-detect Fermi level from OUTCAR
* ``get_structure``        — return CONTCAR or POSCAR as a dict
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import IO, Optional, Tuple, Union

import numpy as np

__all__ = [
    "open_text",
    "open_wavecar",
    "read_ibzkpt_weights",
    "read_oszicar",
    "read_outcar_energy",
    "read_outcar_fermi",
    "read_poscar",
    "get_total_energy",
    "get_fermi_level",
    "get_structure",
]

# ---------------------------------------------------------------------------
# Low-level openers
# ---------------------------------------------------------------------------

_TEXT_OPENERS = {
    ".gz": lambda p: gzip.open(p, "rt", encoding="utf-8"),
    ".bz2": lambda p: bz2.open(p, "rt", encoding="utf-8"),
    ".xz": lambda p: lzma.open(p, "rt", encoding="utf-8"),
    ".lzma": lambda p: lzma.open(p, "rt", encoding="utf-8"),
}

_BIN_OPENERS = {
    ".gz": lambda p: gzip.open(p, "rb"),
    ".bz2": lambda p: bz2.open(p, "rb"),
    ".xz": lambda p: lzma.open(p, "rb"),
    ".lzma": lambda p: lzma.open(p, "rb"),
}


def open_text(path: Union[str, Path]) -> IO[str]:
    """Return a readable text file-like object, decompressing transparently.

    Parameters
    ----------
    path : str or Path
        File path.  Extensions ``.gz``, ``.bz2``, ``.xz``, ``.lzma``
        are decompressed on-the-fly.

    Returns
    -------
    file-like object
        The caller is responsible for closing it (use as a context manager).

    Examples
    --------
    >>> with open_text("IBZKPT.gz") as fh:
    ...     lines = fh.readlines()
    """
    path = Path(path)
    opener = _TEXT_OPENERS.get(path.suffix)
    if opener is not None:
        return opener(path)
    return open(path, "r", encoding="utf-8")


def open_wavecar(
    path: Union[str, Path],
) -> Tuple[IO[bytes], Optional[str]]:
    """Return a seekable binary handle for a WAVECAR, decompressing if needed.

    Because gzip/bz2 streams are not randomly seekable and ``WavecarReader``
    calls ``seek`` extensively, compressed WAVECARs are first decompressed
    to a temporary file.

    Parameters
    ----------
    path : str or Path
        WAVECAR path.  May end in ``.gz``, ``.bz2``, ``.xz``, or ``.lzma``.

    Returns
    -------
    fh : IO[bytes]
        Seekable binary file handle.
    tmp_path : str or None
        Path of the temporary file if decompression was performed; ``None``
        otherwise.  **The caller must delete this file when finished.**

    Examples
    --------
    >>> fh, tmp = open_wavecar("WAVECAR.gz")
    >>> # use fh ...
    >>> fh.close()
    >>> if tmp:
    ...     os.unlink(tmp)
    """
    path = Path(path)
    opener = _BIN_OPENERS.get(path.suffix)
    if opener is None:
        return open(path, "rb"), None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wavecar_tmp")
    os.close(tmp_fd)
    with opener(path) as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return open(tmp_path, "rb"), tmp_path


# ---------------------------------------------------------------------------
# IBZKPT reader
# ---------------------------------------------------------------------------


def read_ibzkpt_weights(ibzkpt_file: Union[str, Path]) -> np.ndarray:
    """Read irreducible k-point weights from a VASP IBZKPT file.

    Supports ``.gz``, ``.bz2``, ``.xz`` compression transparently.

    Parameters
    ----------
    ibzkpt_file : str or Path
        Path to IBZKPT (optionally compressed).

    Returns
    -------
    np.ndarray, shape (nkpts,)
        Raw (unnormalised) k-point weights.

    Examples
    --------
    >>> wts = read_ibzkpt_weights("IBZKPT")
    >>> wts_norm = wts / wts.sum()
    """
    with open_text(ibzkpt_file) as fh:
        lines = fh.readlines()
    weights = []
    for line in lines[3:]:
        line = line.strip()
        if line:
            weights.append(float(line.split()[-1]))
    return np.asarray(weights, dtype=float)


# ---------------------------------------------------------------------------
# OSZICAR reader
# ---------------------------------------------------------------------------


def read_oszicar(oszicar_path: Union[str, Path]) -> dict:
    """Parse a VASP OSZICAR file and return the final electronic step data.

    Parameters
    ----------
    oszicar_path : str or Path
        Path to OSZICAR (optionally compressed).

    Returns
    -------
    dict with keys:

    ``"final_energy"`` : float
        Total energy ``E0`` of the last ionic step (eV).
    ``"free_energy"`` : float
        Free energy ``F`` of the last ionic step (eV).
    ``"steps"`` : list of dict
        Each ionic step: ``{"step": int, "F": float, "E0": float, "dE": float}``.

    Examples
    --------
    >>> data = read_oszicar("OSZICAR")
    >>> E_total = data["final_energy"]
    """
    _FLT = r"[+-]?\d*\.?\d+[Ee][+-]?\d+"
    pattern = re.compile(
        rf"^\s*(\d+)\s+F=\s*({_FLT})\s+E0=\s*({_FLT})\s+d\s*E\s*=\s*({_FLT})"
    )
    steps = []
    with open_text(oszicar_path) as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "F": float(m.group(2)),
                        "E0": float(m.group(3)),
                        "dE": float(m.group(4)),
                    }
                )
    if not steps:
        raise ValueError(f"No ionic steps found in {oszicar_path}")
    return {
        "final_energy": steps[-1]["E0"],
        "free_energy": steps[-1]["F"],
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# OUTCAR readers
# ---------------------------------------------------------------------------


def read_outcar_energy(outcar_path: Union[str, Path]) -> float:
    """Read the final ``energy(sigma->0)`` from OUTCAR.

    Parameters
    ----------
    outcar_path : str or Path

    Returns
    -------
    float
        Total energy in eV.
    """
    pattern = re.compile(r"energy\(sigma->0\)\s*=\s*([+-]?\d+\.\d+)")
    energy = None
    with open_text(outcar_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                energy = float(m.group(1))
    if energy is None:
        raise ValueError(f"'energy(sigma->0)' not found in {outcar_path}")
    return energy


def read_outcar_fermi(outcar_path: Union[str, Path]) -> float:
    """Read the Fermi level from OUTCAR.

    Parameters
    ----------
    outcar_path : str or Path

    Returns
    -------
    float
        Fermi energy in eV.
    """
    pattern = re.compile(r"E-fermi\s*:\s*([+-]?\d+\.\d+)")
    fermi = None
    with open_text(outcar_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                fermi = float(m.group(1))
    if fermi is None:
        raise ValueError(f"'E-fermi' not found in {outcar_path}")
    return fermi


# ---------------------------------------------------------------------------
# POSCAR/CONTCAR minimal parser
# ---------------------------------------------------------------------------

# Atomic masses for common elements (g/mol)
_ATOMIC_MASS: dict = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.96,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.904,
    "Xe": 131.293,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.242,
    "Hf": 178.49,
    "Ta": 180.948,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.967,
    "Hg": 200.592,
    "Tl": 204.383,
    "Pb": 207.2,
    "Bi": 208.980,
}


def read_poscar(poscar_path: Union[str, Path]) -> dict:
    """Minimal POSCAR/CONTCAR parser.

    Supports selective dynamics, Direct/Cartesian coordinates, and
    compressed files.

    Parameters
    ----------
    poscar_path : str or Path

    Returns
    -------
    dict with keys:

    ``"title"`` : str
    ``"scale"`` : float
    ``"lattice"`` : np.ndarray, shape (3, 3)  — lattice vectors (Å)
    ``"species"`` : list of str
    ``"counts"`` : list of int
    ``"atom_species"`` : list of str  — per-atom species labels
    ``"positions"`` : np.ndarray, shape (natoms, 3)  — fractional coordinates
    ``"masses"`` : list of float  — atomic masses (g/mol)
    ``"natoms"`` : int

    Examples
    --------
    >>> s = read_poscar("POSCAR")
    >>> lattice = s["lattice"]
    """
    with open_text(poscar_path) as fh:
        raw = fh.readlines()

    lines = [line.rstrip("\n") for line in raw]
    title = lines[0]
    scale = float(lines[1].strip())
    lattice = np.array([lines[i].split() for i in range(2, 5)], dtype=float) * scale

    species_or_counts = lines[5].split()
    if species_or_counts[0].isalpha():
        species = species_or_counts
        counts = list(map(int, lines[6].split()))
        coord_start = 7
    else:
        species = [f"X{i}" for i in range(len(species_or_counts))]
        counts = list(map(int, species_or_counts))
        coord_start = 6

    natoms = sum(counts)
    coord_line = lines[coord_start].strip().upper()
    if coord_line.startswith("S"):
        coord_start += 1
        coord_line = lines[coord_start].strip().upper()

    is_direct = coord_line.startswith("D")

    positions = []
    for i in range(coord_start + 1, coord_start + 1 + natoms):
        vals = lines[i].split()[:3]
        positions.append([float(v) for v in vals])
    positions = np.array(positions)

    if not is_direct:
        positions = np.dot(positions, np.linalg.inv(lattice))

    atom_species = []
    masses = []
    for sp, count in zip(species, counts):
        for _ in range(count):
            atom_species.append(sp)
            masses.append(_ATOMIC_MASS.get(sp, 1.0))

    return {
        "title": title,
        "scale": scale,
        "lattice": lattice,
        "species": species,
        "counts": counts,
        "atom_species": atom_species,
        "positions": positions,
        "masses": masses,
        "natoms": natoms,
    }


# ---------------------------------------------------------------------------
# Auto-detect total energy / Fermi level from a calculation directory
# ---------------------------------------------------------------------------


def get_total_energy(
    calc_dir: Union[str, Path],
    prefer: str = "oszicar",
) -> float:
    """Read the total energy from a VASP calculation directory.

    Searches (in priority order) for OSZICAR, OUTCAR, or vasprun.xml.

    Parameters
    ----------
    calc_dir : str or Path
        Directory containing VASP outputs.
    prefer : {'oszicar', 'outcar', 'vasprun'}
        Which file to try first.

    Returns
    -------
    float
        Total energy in eV.

    Examples
    --------
    >>> E = get_total_energy("/data/ground_state/")
    """
    d = Path(calc_dir)
    candidates = {
        "oszicar": [d / "OSZICAR", d / "OSZICAR.gz", d / "OSZICAR.bz2"],
        "outcar": [d / "OUTCAR", d / "OUTCAR.gz", d / "OUTCAR.bz2"],
        "vasprun": [d / "vasprun.xml", d / "vasprun.xml.gz"],
    }
    order = (
        ["oszicar", "outcar", "vasprun"]
        if prefer == "oszicar"
        else ["outcar", "oszicar", "vasprun"]
        if prefer == "outcar"
        else ["vasprun", "oszicar", "outcar"]
    )
    for key in order:
        for path in candidates[key]:
            if path.exists():
                try:
                    if key == "oszicar":
                        return read_oszicar(path)["final_energy"]
                    elif key == "outcar":
                        return read_outcar_energy(path)
                    else:
                        from pymatgen.io.vasp.outputs import Vasprun

                        return Vasprun(str(path)).final_energy
                except Exception:
                    continue
    raise FileNotFoundError(
        f"Could not find or parse OSZICAR/OUTCAR/vasprun.xml in {calc_dir}"
    )


def get_fermi_level(calc_dir: Union[str, Path]) -> float:
    """Read the Fermi level from OUTCAR in a VASP calculation directory.

    Parameters
    ----------
    calc_dir : str or Path

    Returns
    -------
    float
        Fermi energy in eV.

    Examples
    --------
    >>> ef = get_fermi_level("/data/ground_state/")
    """
    d = Path(calc_dir)
    for fname in ["OUTCAR", "OUTCAR.gz", "OUTCAR.bz2"]:
        p = d / fname
        if p.exists():
            try:
                return read_outcar_fermi(p)
            except Exception:
                continue
    for fname in ["vasprun.xml", "vasprun.xml.gz"]:
        p = d / fname
        if p.exists():
            try:
                from pymatgen.io.vasp.outputs import Vasprun

                return Vasprun(str(p)).efermi
            except Exception:
                continue
    raise FileNotFoundError(f"Could not determine Fermi level from files in {calc_dir}")


def get_structure(calc_dir: Union[str, Path], relaxed: bool = True) -> dict:
    """Return structure from CONTCAR (relaxed) or POSCAR in a directory.

    Parameters
    ----------
    calc_dir : str or Path
    relaxed : bool
        If True, prefer CONTCAR; else POSCAR.

    Returns
    -------
    dict from :func:`read_poscar`.

    Examples
    --------
    >>> s = get_structure("/data/ground/")
    >>> lattice = s["lattice"]
    """
    d = Path(calc_dir)
    search_order = ["CONTCAR", "POSCAR"] if relaxed else ["POSCAR", "CONTCAR"]
    for fname in search_order:
        for ext in ["", ".gz", ".bz2"]:
            p = d / (fname + ext)
            if p.exists() and p.stat().st_size > 0:
                return read_poscar(p)
    raise FileNotFoundError(f"No POSCAR/CONTCAR found in {calc_dir}")
