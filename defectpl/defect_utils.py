"""
defectpl.defect_utils
=====================

Utilities for generating the prerequisite JSON files needed by the
participation ratio module — without requiring pydefect.

Functions
---------
make_defect_entry
    Build a ``defect_entry.json`` from a defect name and centre coordinates
    (manual) or by comparing perfect and defect structures (auto-detect).

make_defect_structure_info
    Build a ``defect_structure_info.json`` by running a distance-based
    neighbour search on a POSCAR/CONTCAR file.

detect_defect_center
    Detect the defect centre (fractional coords) by comparing a perfect
    and defect structure — supports vacancies, interstitials, and
    substitutions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFECT_ENTRY_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Defect centre auto-detection
# ---------------------------------------------------------------------------


def detect_defect_center(
    perfect_poscar: str | Path,
    defect_poscar: str | Path,
    site_tol: float = 0.5,
) -> Tuple[List[float], str]:
    """
    Detect the defect centre by comparing a perfect and defect structure.

    Works for vacancies (site in perfect but not in defect) and
    interstitials/substitutions (site in defect but not in perfect).

    Parameters
    ----------
    perfect_poscar : str or Path
        POSCAR/CONTCAR for the perfect (pristine) supercell.
    defect_poscar : str or Path
        POSCAR/CONTCAR for the defect supercell.
    site_tol : float, optional
        Cartesian distance tolerance (Å) for matching sites.  Default 0.5 Å.

    Returns
    -------
    (center_frac, defect_type)
        ``center_frac`` — fractional coordinates [x, y, z].
        ``defect_type`` — one of ``"vacancy"``, ``"interstitial"``,
        ``"substitution"``, or ``"unknown"``.

    Raises
    ------
    ValueError
        If the two structures have incompatible lattices or if auto-detection
        fails unambiguously.
    ImportError
        If pymatgen is not installed.
    """
    from pymatgen.core import Structure

    perfect = Structure.from_file(str(perfect_poscar))
    defect = Structure.from_file(str(defect_poscar))

    p_cart = np.array([s.coords for s in perfect])
    d_cart = np.array([s.coords for s in defect])

    # Find sites in perfect that have no match in defect → vacancy sites
    missing_in_defect: List[int] = []
    for ip, pc in enumerate(p_cart):
        dists = np.linalg.norm(d_cart - pc, axis=1)
        if dists.min() > site_tol:
            missing_in_defect.append(ip)

    # Find sites in defect that have no match in perfect → interstitial/sub sites
    extra_in_defect: List[int] = []
    for id_, dc in enumerate(d_cart):
        dists = np.linalg.norm(p_cart - dc, axis=1)
        if dists.min() > site_tol:
            extra_in_defect.append(id_)

    n_missing = len(missing_in_defect)
    n_extra = len(extra_in_defect)

    if n_missing == 1 and n_extra == 0:
        # Vacancy: centre is the removed atom's position in the perfect cell
        frac = perfect[missing_in_defect[0]].frac_coords.tolist()
        return frac, "vacancy"

    if n_missing == 0 and n_extra == 1:
        # Interstitial: centre is the new atom's position in the defect cell
        frac = defect[extra_in_defect[0]].frac_coords.tolist()
        return frac, "interstitial"

    if n_missing == 1 and n_extra == 1:
        # Substitution: use the position of the replaced site (perfect cell)
        frac = perfect[missing_in_defect[0]].frac_coords.tolist()
        return frac, "substitution"

    if n_missing > 0:
        # Multiple vacancies — take centroid of missing sites
        coords = np.array([perfect[i].frac_coords for i in missing_in_defect])
        center = coords.mean(axis=0)
        # Wrap into [0, 1)
        center = center % 1.0
        logger.warning(
            "Multiple missing sites (%d); using centroid as defect centre.", n_missing
        )
        return center.tolist(), "unknown"

    raise ValueError(
        f"Could not auto-detect defect centre: {n_missing} missing sites, "
        f"{n_extra} extra sites. Provide --center manually."
    )


# ---------------------------------------------------------------------------
# make_defect_entry
# ---------------------------------------------------------------------------


def make_defect_entry(
    name: str,
    center: Optional[Sequence[float]] = None,
    perfect_poscar: Optional[str | Path] = None,
    defect_poscar: Optional[str | Path] = None,
    out_path: str | Path = "defect_entry.json",
    site_tol: float = 0.5,
) -> dict:
    """
    Create a ``defect_entry.json`` file.

    Either supply ``center`` directly, or let the function auto-detect it by
    comparing ``perfect_poscar`` and ``defect_poscar``.

    Parameters
    ----------
    name : str
        Defect label (e.g. ``"Va_O1_2"`` for an oxygen vacancy with charge +2).
    center : sequence of 3 floats, optional
        Fractional coordinates of the defect centre.  Required when
        ``perfect_poscar`` / ``defect_poscar`` are not provided.
    perfect_poscar : str or Path, optional
        Path to the perfect (undoped) supercell POSCAR/CONTCAR.
    defect_poscar : str or Path, optional
        Path to the defect supercell POSCAR/CONTCAR.
    out_path : str or Path, optional
        Destination for the JSON file.  Default ``"defect_entry.json"``.
    site_tol : float, optional
        Cartesian tolerance (Å) for matching sites in auto-detection.

    Returns
    -------
    dict
        The JSON content that was written.

    Raises
    ------
    ValueError
        If neither ``center`` nor both structure files are provided.
    """
    if center is not None:
        defect_center = list(float(x) for x in center)
        defect_type = "manual"
    elif perfect_poscar is not None and defect_poscar is not None:
        defect_center, defect_type = detect_defect_center(
            perfect_poscar, defect_poscar, site_tol=site_tol
        )
        logger.info(
            "Auto-detected defect type '%s', centre: %s", defect_type, defect_center
        )
    else:
        raise ValueError(
            "Provide either --center or both --perfect and --defect structure files."
        )

    payload = {
        "schema_version": DEFECT_ENTRY_SCHEMA_VERSION,
        "name": name,
        "defect_center": defect_center,
        "defect_type": defect_type,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("defect_entry.json written: %s", out_path)
    return payload


# ---------------------------------------------------------------------------
# make_defect_structure_info
# ---------------------------------------------------------------------------


def make_defect_structure_info(
    poscar: str | Path,
    defect_center_frac: Sequence[float],
    cutoff_radius: float = 3.5,
    out_path: str | Path = "defect_structure_info.json",
) -> dict:
    """
    Generate a ``defect_structure_info.json`` via distance-based neighbour search.

    Finds all atoms within *cutoff_radius* Å of *defect_center_frac* in the
    supercell defined by *poscar* (minimum-image convention).

    Parameters
    ----------
    poscar : str or Path
        POSCAR / CONTCAR of the defect supercell.
    defect_center_frac : sequence of 3 floats
        Fractional coordinates of the defect centre.
    cutoff_radius : float, optional
        Search radius in Å.  Default 3.5 Å.
    out_path : str or Path, optional
        Destination file.  Default ``"defect_structure_info.json"``.

    Returns
    -------
    dict
        The JSON content that was written, with key ``"neighbor_atom_indices"``.

    Raises
    ------
    ImportError
        If pymatgen is not installed.
    """
    from pymatgen.core import Structure

    poscar = Path(poscar)
    struct = Structure.from_file(str(poscar))
    center = np.array(defect_center_frac, dtype=float)

    neighbors: List[dict] = []
    for i, site in enumerate(struct):
        diff = center - site.frac_coords
        diff -= np.round(diff)  # minimum image
        d = float(np.linalg.norm(struct.lattice.get_cartesian_coords(diff)))
        if d < cutoff_radius:
            neighbors.append(
                {"index": i, "element": str(site.specie), "distance": round(d, 5)}
            )

    indices = [n["index"] for n in neighbors]

    payload = {
        "poscar": str(poscar),
        "defect_center_frac": list(float(x) for x in defect_center_frac),
        "cutoff_radius": cutoff_radius,
        "n_neighbors": len(indices),
        "neighbor_atom_indices": indices,
        "neighbors": neighbors,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "defect_structure_info.json written: %s  (%d neighbours)",
        out_path,
        len(indices),
    )
    return payload


# ---------------------------------------------------------------------------
# Utility: parse fractional coordinate string  "x,y,z"  or  "x y z"
# ---------------------------------------------------------------------------


def parse_frac_coords(text: str) -> List[float]:
    """
    Parse a fractional-coordinate string into a list of 3 floats.

    Accepts either comma-separated (``"0.5,0.5,0.5"``) or
    space-separated (``"0.5 0.5 0.5"``) formats.

    Parameters
    ----------
    text : str

    Returns
    -------
    list of 3 floats

    Raises
    ------
    ValueError
        If exactly 3 values cannot be parsed.
    """
    import re

    parts = re.split(r"[,\s]+", text.strip())
    try:
        coords = [float(p) for p in parts if p]
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse fractional coordinates from '{text}': {exc}"
        ) from exc
    if len(coords) != 3:
        raise ValueError(
            f"Expected 3 fractional coordinates, got {len(coords)} from '{text}'"
        )
    return coords
