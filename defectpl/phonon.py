# -*- coding: utf-8 -*-
"""
Module for parsing, calculating, and managing Gamma-point phonon properties 
using Phonopy and VASP outputs. All calculations convert frequency data to eV.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import yaml

from monty.json import MSONable
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_SETS, parse_FORCE_CONSTANTS
from phonopy.interface.vasp import create_FORCE_CONSTANTS, read_vasp

# Import energy rescaling metric conversion factors from constants
from defectpl.constants import THZ2EV


class GammaPhononData(MSONable):
    """
    An MSONable data container storing processed Gamma-point phonon properties.

    Parameters
    ----------
    frequencies : list of float
        Vibrational mode frequencies at the Gamma point, rescaled to eV.
    eigenvectors : list of list of float
        Real-component displacement eigenvectors matrix of shape (nmodes, natoms * 3).
    masses : list of float
        Atomic masses per active crystal species site coordinate.
    natoms : int
        Total number of internal core atoms embedded in the structural cell.
    nmodes : int
        Total count of normal vibrational mode pathways ($3 \\times N$).
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
) -> Dict[str, Any]:
    """
    Calculate irreducible representations (irreps) of phonon modes at the Gamma point
    and export the computed symmetry details to a Phonopy YAML format.

    Parameters
    ----------
    unitcell_path : str or pathlib.Path
        Path to the unit cell VASP structural input (e.g., POSCAR).
    force_constants_path : str or pathlib.Path, optional
        Path to a pre-calculated FORCE_CONSTANTS file.
    force_sets_path : str or pathlib.Path, optional
        Path to a FORCE_SETS file if constants need parsing from displacements.
    dimension : str, list of int, or numpy.ndarray, default "1 1 1"
        Supercell matrix dimension layout configuration array.
    symprec : float, default 1e-5
        Symmetry determination spatial tolerance window tracking crystal sites.
    degeneracy_tolerance : float, optional
        Tolerance scale used to group closely degenerate paths into unique irreps.
    nac_q_direction : list of float, optional
        Modulation direction path mapping Non-Analytical Corrections at $q=0$.
    is_little_cogroup : bool, default False
        If True, enforces symmetry tracking matching little co-group mechanics rules.

    Returns
    -------
    dict
        A parsed representation summary dictionary mapping rotation symbols and 
        real-casted characters data layers.
    """
    if isinstance(dimension, str):
        dim = np.array([int(x) for x in dimension.split()])
    else:
        dim = np.array(dimension)

    unitcell = read_vasp(str(unitcell_path))
    phonon = Phonopy(unitcell, supercell_matrix=np.diag(dim), symprec=symprec)

    if force_constants_path is not None:
        phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants_path))
    elif force_sets_path is not None:
        phonon.dataset = parse_FORCE_SETS(filename=str(force_sets_path))
        phonon.produce_force_constants()
    else:
        raise ValueError("Either force_constants_path or force_sets_path must be provided.")

    # 1. Execute Irreducible Representations mapping engine on Gamma point
    phonon.set_irreps(
        q=[0, 0, 0],
        is_little_cogroup=is_little_cogroup,
        nac_q_direction=nac_q_direction,
        degeneracy_tolerance=degeneracy_tolerance,
    )

    # 2. Safely export the underlying symmetry profile using the irreps writer engine
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
        Path to the reference crystal unit cell template file.
    force_constants_filename : str or pathlib.Path, default "FORCE_CONSTANTS"
        Path to the parsed force constants array file.
    dimension : str, list of int, or numpy.ndarray, default "1 1 1"
        Supercell scaling matrix boundaries expansion configurations.
    symprec : float, default 1e-5
        Symmetry evaluation threshold tracking lattice matching limits.
    output_filename : str or pathlib.Path, default "band.yaml"
        Destination output path where the compiled Phonopy YAML structure will be dropped.
    """
    if isinstance(dimension, str):
        dim = np.array([int(x) for x in dimension.split()])
    else:
        dim = np.array(dimension)

    unitcell = read_vasp(str(unitcell_filename))
    phonon = Phonopy(unitcell, supercell_matrix=np.diag(dim), symprec=symprec)
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants_filename))

    bands = [[[0, 0, 0], [0, 0, 0]]]
    phonon.run_band_structure(bands)
    phonon.write_yaml_band_structure(filename=str(output_filename))


def read_band_yaml(band_yaml: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Parse a Phonopy band.yaml file to extract frequencies (in eV), eigenvectors, and atomic masses.

    Parameters
    ----------
    band_yaml : str or pathlib.Path
        The path tracking the file destination for an active `band.yaml` profile.

    Returns
    -------
    frequencies_converted : numpy.ndarray
        1D array of imaginary-clipped phonon frequencies converted to eV units.
    geigenvecs : numpy.ndarray
        Real-component displacement matrix tracking normalized atomic coordinates.
    masses_list : list of float
        The extracted atomic mass sequence mapped to crystal index indices.
    """
    path = Path(band_yaml)
    if not path.exists():
        raise FileNotFoundError(f"Phonopy band configuration file missing at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        band_data = yaml.safe_load(f)

    q_idx = 0
    phonon_modes = band_data["phonon"][q_idx]["band"]

    frequencies_raw = np.array([mode["frequency"] for mode in phonon_modes], dtype=float)
    geigenvecs = np.array([mode["eigenvector"] for mode in phonon_modes], dtype=float)
    
    if geigenvecs.ndim > 2:
        geigenvecs = geigenvecs[..., 0]

    masses_list = [point["mass"] for point in band_data["points"]]

    # Clip negative/imaginary frequencies and convert from THz to eV
    frequencies_raw[frequencies_raw < 0.0] = 0.0
    frequencies_converted = frequencies_raw * THZ2EV

    return frequencies_converted, geigenvecs, masses_list


def extract_gamma_phonon_data(band_yaml_path: Union[str, Path]) -> GammaPhononData:
    """
    High-level factory function to extract and instantiate a GammaPhononData container from a band.yaml file.

    Parameters
    ----------
    band_yaml_path : str or pathlib.Path
        Target input phonopy band calculation configuration tracker.

    Returns
    -------
    GammaPhononData
        An initialized MSONable dataclass containing rescaled parameters.
    """
    freqs, evecs, masses = read_band_yaml(band_yaml_path)
    natoms = len(masses)
    nmodes = len(freqs)
    
    return GammaPhononData(
        frequencies=freqs.tolist(),
        eigenvectors=evecs.tolist(),
        masses=masses,
        natoms=natoms,
        nmodes=nmodes,
        meta_info={"source_file": str(band_yaml_path)}
    )