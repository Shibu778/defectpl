# -*- coding: utf-8 -*-
"""
Module for parsing, calculating, and managing Gamma-point phonon properties
using Phonopy and VASP outputs. All calculations convert frequency data to eV.

Author: Shibu Meher
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from monty.json import MSONable

# Import energy rescaling metric conversion factors from constants
from defectpl.constants import THZ2EV


# phonopy is a declared dependency but may not be present in all environments.
# All phonopy symbols are imported lazily inside the functions that need them so
# that the rest of this module (GammaPhononData, read_band_yaml, etc.) works
# without it.
def _require_phonopy(fn_name: str = "") -> None:
    """Raise a clear error when phonopy is missing."""
    try:
        import phonopy  # noqa: F401
    except ImportError as exc:
        msg = "phonopy is required for phonon calculations"
        if fn_name:
            msg += f" (needed by '{fn_name}')"
        msg += ".  Install with:  pip install phonopy"
        raise ImportError(msg) from exc


class GammaPhononData(MSONable):
    """
    An MSONable data container storing processed Gamma-point phonon properties.

    Parameters
    ----------
    frequencies : list of float
        Vibrational mode frequencies at the Gamma point, rescaled to eV.
    eigenvectors : list of list of list of float
        Real-component displacement eigenvectors of shape (nmodes, natoms, 3).
    masses : list of float
        Atomic masses per active crystal species site coordinate.
    natoms : int
        Total number of internal core atoms embedded in the structural cell.
    nmodes : int
        Total count of normal vibrational mode pathways (3 x N).
    meta_info : dict, optional
        Metadata tracking runtime settings, dimensions, or target accuracy.
    """

    def __init__(
        self,
        frequencies: List[float],
        eigenvectors: List[List[float]],
        masses: List[float],
        natoms: int,
        nmodes: int,
        meta_info: Optional[Dict[str, Any]] = None,
    ):
        self.frequencies = frequencies
        self.eigenvectors = eigenvectors
        self.masses = masses
        self.natoms = natoms
        self.nmodes = nmodes
        self.meta_info = meta_info or {}

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the instance properties to a JSON-compatible dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "frequencies": self.frequencies,
            "eigenvectors": self.eigenvectors,
            "masses": self.masses,
            "natoms": self.natoms,
            "nmodes": self.nmodes,
            "meta_info": self.meta_info,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GammaPhononData":
        """Reconstruct a class instance from an unpacked serialization dictionary."""
        return cls(
            frequencies=d["frequencies"],
            eigenvectors=d["eigenvectors"],
            masses=d["masses"],
            natoms=d["natoms"],
            nmodes=d["nmodes"],
            meta_info=d.get("meta_info", {}),
        )


def create_force_constants_from_vasprun(
    vasprun_filename: Union[str, Path], is_hdf5: bool = False, log_level: int = 1
) -> int:
    """
    Parse a vasprun.xml file and write extracted force constants to disk.

    Parameters
    ----------
    vasprun_filename : str or pathlib.Path
        Path to the target input vasprun.xml file.
    is_hdf5 : bool, default False
        If True, writes to force_constants.hdf5; otherwise writes to FORCE_CONSTANTS.
    log_level : int, default 1
        Phonopy verbosity output log level.

    Returns
    -------
    int
        0 on execution success, 1 on execution failure.
    """
    _require_phonopy("create_force_constants_from_vasprun")
    from phonopy.interface.vasp import create_FORCE_CONSTANTS

    return create_FORCE_CONSTANTS(str(vasprun_filename), is_hdf5, log_level)


def calculate_phonon_symmetries(
    unitcell_path: Union[str, Path],
    force_constants_path: Optional[Union[str, Path]] = None,
    force_sets_path: Optional[Union[str, Path]] = None,
    dimension: Union[str, List[int], np.ndarray] = "1 1 1",
    symprec: float = 1e-5,
    degeneracy_tolerance: Optional[float] = None,
    nac_q_direction: Optional[List[float]] = None,
    is_little_cogroup: bool = False,
) -> None:
    """
    Calculate irreducible representations (irreps) of phonon modes at the Gamma point
    and export the computed symmetry details to a Phonopy YAML format.

    Parameters
    ----------
    unitcell_path : str or pathlib.Path
        Path to a structural VASP input unitcell geometry file (POSCAR).
    force_constants_path : str or pathlib.Path, optional
        Path pointing to an active FORCE_CONSTANTS file layout.
    force_sets_path : str or pathlib.Path, optional
        Path pointing to an active FORCE_SETS file layout.
    dimension : str, list of int, or numpy.ndarray, default "1 1 1"
        The expansion dimensions matrix configuring supercell construction loops.
    symprec : float, default 1e-5
        Distance tolerance metric required to map equivalent atomic positions.
    degeneracy_tolerance : float, optional
        Energy cutoff width identifying degenerate frequency bands.
    nac_q_direction : list of float, optional
        The q-vector direction for non-analytical term corrections (NAC).
    is_little_cogroup : bool, default False
        Determines group representation parsing parameters.
    """
    _require_phonopy("calculate_phonon_symmetries")
    from phonopy import Phonopy
    from phonopy.file_IO import parse_FORCE_CONSTANTS, parse_FORCE_SETS
    from phonopy.interface.vasp import read_vasp

    if isinstance(dimension, str):
        dim = np.array([int(x) for x in dimension.split()])
    else:
        dim = np.array(dimension)

    unitcell = read_vasp(str(unitcell_path))
    phonon = Phonopy(unitcell, supercell_matrix=np.diag(dim), symprec=symprec)

    if force_constants_path is not None:
        phonon.force_constants = parse_FORCE_CONSTANTS(
            filename=str(force_constants_path)
        )
    elif force_sets_path is not None:
        phonon.dataset = parse_FORCE_SETS(filename=str(force_sets_path))
        phonon.produce_force_constants()
    else:
        raise ValueError(
            "Either force_constants_path or force_sets_path must be provided."
        )

    phonon.set_irreps(
        q=[0, 0, 0],
        is_little_cogroup=is_little_cogroup,
        nac_q_direction=nac_q_direction,
        degeneracy_tolerance=degeneracy_tolerance,
    )

    phonon.write_yaml_irreps()


def calculate_gamma_phonon_to_band_yaml(
    unitcell_filename: Union[str, Path] = "POSCAR",
    force_constants_filename: Union[str, Path] = "FORCE_CONSTANTS",
    dimension: Union[str, List[int], np.ndarray] = "1 1 1",
    symprec: float = 1e-5,
    output_filename: Union[str, Path] = "band.yaml",
) -> None:
    """
    Evaluate phonon modes at the Gamma point using force constants and write to a band.yaml file.

    Parameters
    ----------
    unitcell_filename : str or pathlib.Path, default "POSCAR"
        Path to the primary reference cell geometry description file.
    force_constants_filename : str or pathlib.Path, default "FORCE_CONSTANTS"
        Path addressing source file storing parsed 2nd-order derivatives.
    dimension : str, list of int, or numpy.ndarray, default "1 1 1"
        The structural supercell replication shape vector configuration array.
    symprec : float, default 1e-5
        Structural crystal space group symmetry parsing threshold tolerance.
    output_filename : str or pathlib.Path, default "band.yaml"
        Destination filename target for generating the Phonopy track output data file.
    """
    _require_phonopy("calculate_gamma_phonon_to_band_yaml")
    from phonopy import Phonopy
    from phonopy.file_IO import parse_FORCE_CONSTANTS
    from phonopy.interface.vasp import read_vasp

    if isinstance(dimension, str):
        dim = np.array([int(x) for x in dimension.split()])
    else:
        dim = np.array(dimension)

    unitcell = read_vasp(str(unitcell_filename))
    phonon = Phonopy(unitcell, supercell_matrix=np.diag(dim), symprec=symprec)
    phonon.force_constants = parse_FORCE_CONSTANTS(
        filename=str(force_constants_filename)
    )

    bands = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    phonon.run_band_structure(bands, with_eigenvectors=True)
    phonon.write_yaml_band_structure(filename=str(output_filename))


def read_band_yaml(
    band_yaml_path: Union[str, Path],
    q_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses a Phonopy band.yaml summary output file to extract Gamma-point
    phonon frequencies, displacement eigenvectors, and atomic masses.

    Parameters
    ----------
    band_yaml_path : str or pathlib.Path
        The filename tracking location of the processed Phonopy yaml payload.
    q_idx : int, default 0
        The absolute entry loop lookup selection index focusing on a specific q-point.

    Returns
    -------
    frequencies : numpy.ndarray
        A flat array of all parsed vibrational mode frequencies, rescaled to eV.
    eigenvectors : numpy.ndarray
        Real displacement eigenvectors of shape (nmodes, natoms, 3).
    masses : numpy.ndarray
        Array containing mass metrics for each ion matching index layout configurations.
    """
    with open(str(band_yaml_path), "r") as f:
        band = yaml.safe_load(f)

    n_atoms = band["natom"]
    nmodes = len(band["phonon"][q_idx]["band"])

    # 1. Parse frequencies and safely bound acoustic/imaginary noise to 0.0
    gfrequencies = np.array(
        [band["phonon"][q_idx]["band"][i]["frequency"] for i in range(nmodes)],
        dtype=float,
    )
    gfrequencies[gfrequencies < 0.0] = 0.0
    gfrequencies *= THZ2EV  # Convert THz -> eV

    # 2. Parse eigenvectors (shape from yaml is often (nmodes, natoms, 3, 2))
    geigenvecs = np.array(
        [band["phonon"][q_idx]["band"][i]["eigenvector"] for i in range(nmodes)],
        dtype=complex,
    )
    # Strip complex phase: (nmodes, natoms, 3, 2) -> (nmodes, natoms, 3)
    eigenvectors = np.array(geigenvecs[..., 0].real, dtype=float)

    # 3. Gather masses
    masses = np.asarray(
        [band["points"][i]["mass"] for i in range(n_atoms)], dtype=float
    )

    return gfrequencies, eigenvectors, masses


def extract_gamma_phonon_data(band_yaml_path: Union[str, Path]) -> GammaPhononData:
    """
    High-level factory function to extract and instantiate a GammaPhononData container from a band.yaml file.

    Parameters
    ----------
    band_yaml_path : str or pathlib.Path
        The destination track file path addressing parsed band structure information.

    Returns
    -------
    GammaPhononData
        An operational, JSON-serializable structured database holding phononic properties.
    """
    freqs, evecs, masses = read_band_yaml(band_yaml_path, q_idx=0)
    natoms = len(masses)
    nmodes = len(freqs)

    return GammaPhononData(
        frequencies=freqs.tolist(),
        eigenvectors=evecs.tolist(),
        masses=masses.tolist(),
        natoms=natoms,
        nmodes=nmodes,
        meta_info={"source_file": str(band_yaml_path)},
    )
