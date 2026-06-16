# -*- coding: utf-8 -*-
"""
VASP-specific I/O layer for defectpl.

Consolidates all VASP file parsing and workflow helpers.  All
pymatgen imports are **lazy** (inside function bodies) so this module can
be imported without pymatgen installed.

Public surface
--------------
Eigenvalue / electronic
    get_spin_multiplicity, read_eigenval_file

OUTCAR / trajectory
    check_outcar_convergence, get_nions, get_species_and_index_map,
    get_structures_and_forces, get_final_structure_and_forces_from_outcar,
    get_first_structure_and_forces_from_outcar, OutcarParser

PL workflow helpers
    calc_dF, prepare_dF_files, calc_dR, calc_delta_Q, get_q_from_structure,
    generate_ccd_calculations, analyze_ccd_framework,
    run_pl_calc_vasp_displacement_mode, run_pl_calc_vasp_force_mode,
    run_dynamic_yaml_comparison, run_kohn_sham_analysis

VaspReader
    Implements :class:`~defectpl.io.base.PhononReader` and
    :class:`~defectpl.io.base.ElectronicReader` protocols.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from defectpl.utils import calc_delQ

if TYPE_CHECKING:
    from pymatgen.core import Structure


# =====================================================================
# Kohn-Sham Eigenvalue & Electronic Parsing
# =====================================================================


def get_spin_multiplicity(homo_up_idx: int, homo_down_idx: int) -> float:
    """Calculate the spin multiplicity (2S+1) from HOMO level indices."""
    S = abs(homo_up_idx - homo_down_idx) / 2.0
    return 2.0 * S + 1.0


def read_eigenval_file(filename: Union[str, Path], k_idx: int = 0) -> Dict[str, Any]:
    """
    Parse a VASP EIGENVAL file and return spin-resolved eigenvalues at one k-point.

    Parameters
    ----------
    filename : str or Path
    k_idx : int
        Zero-based k-point index (default 0, i.e. Γ-point).

    Returns
    -------
    dict  — see :func:`defectpl.vasp.read_eigenval_file` for full key list.

    Raises
    ------
    ValueError
        If the calculation is not spin-polarised (ISPIN ≠ 2).
    """
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp.outputs import Eigenval

    from defectpl.ks_analysis import get_homo_lumo_idx

    data: Dict[str, Any] = {}
    eig = Eigenval(filename, separate_spins=True)
    if eig.ispin != 2:
        raise ValueError("The calculation is not spin polarized.")

    print(f"Selecting the {k_idx}-th k-point from {eig.nkpt} k-points.")
    print(f"Selected k-point: {eig.kpoints[k_idx]}")

    data["up"] = list(eig.eigenvalues[Spin.up][k_idx])
    data["down"] = list(eig.eigenvalues[Spin.down][k_idx])

    data["homo_up_idx"], data["lumo_up_idx"] = get_homo_lumo_idx(data["up"])
    data["homo_down_idx"], data["lumo_down_idx"] = get_homo_lumo_idx(data["down"])

    data["homo_up"] = eig.eigenvalue_band_properties[2][0]
    data["homo_down"] = eig.eigenvalue_band_properties[2][1]
    data["lumo_up"] = eig.eigenvalue_band_properties[1][0]
    data["lumo_down"] = eig.eigenvalue_band_properties[1][1]
    data["hl_gap_up"] = eig.eigenvalue_band_properties[0][0]
    data["hl_gap_down"] = eig.eigenvalue_band_properties[0][1]

    data["nelect"] = eig.nelect
    data["nbands"] = eig.nbands
    data["nkpt"] = eig.nkpt
    data["selected_kpoint"] = [k_idx, list(eig.kpoints[k_idx])]
    data["spin_multiplicity"] = get_spin_multiplicity(
        data["homo_up_idx"], data["homo_down_idx"]
    )
    return data


# =====================================================================
# OUTCAR / Trajectory Parsing
# =====================================================================


def check_outcar_convergence(outcar_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Check electronic and structural convergence from a VASP OUTCAR file.

    Returns
    -------
    dict with keys ``structural_converged``, ``electronic_converged``,
    ``finished_cleanly``.
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    results = {
        "structural_converged": False,
        "electronic_converged": True,
        "finished_cleanly": False,
    }

    tail_lines: deque = deque(maxlen=100)
    has_content = False

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            has_content = True
            tail_lines.append(line)
            if "ELECTRONIC CONVERGENCE MINIMIZATION" in line and "not achieved" in line:
                results["electronic_converged"] = False

    if not has_content:
        return {k: False for k in results}

    for line in tail_lines:
        if "reached required accuracy" in line:
            results["structural_converged"] = True
        tokens = line.split()
        if len(tokens) >= 2 and ("User" in tokens[0] and "time" in tokens[1]):
            results["finished_cleanly"] = True
        elif "Total CPU time" in line:
            results["finished_cleanly"] = True

    return results


def get_nions(outcar_path: Union[str, Path]) -> int:
    """Extract the NIONS count from a VASP OUTCAR file."""
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "NIONS =" in line:
                return int(line.split("=")[-1])

    raise ValueError(f"Could not find 'NIONS =' token within {outcar_path}")


def get_species_and_index_map(outcar_path: Union[str, Path]) -> List[str]:
    """
    Build a flat per-atom element list from OUTCAR POTCAR entries.

    Handles multi-occurrence POTCAR definitions (e.g. N, C, N, C) by
    aligning to the 'ions per type' array.
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    species_types: List[str] = []
    ions_per_type: List[int] = []

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "POTCAR:" in line and "PAW" in line:
                tokens = line.split()
                if len(tokens) >= 3:
                    species_types.append(tokens[2].split("_")[0])
            if "ions per type =" in line:
                ions_per_type = [int(x) for x in line.split("=")[-1].split()]
                break

    if not species_types or not ions_per_type:
        raise ValueError(
            f"Could not fully parse species maps from {outcar_path}. "
            f"Found species types: {species_types}, Counts: {ions_per_type}"
        )

    if len(species_types) < len(ions_per_type):
        raise ValueError(
            f"OUTCAR processing error: Parsed fewer POTCAR species entries "
            f"({len(species_types)}) than blocks in 'ions per type' ({len(ions_per_type)})."
        )

    species_types = species_types[: len(ions_per_type)]
    species_map: List[str] = []
    for element, count in zip(species_types, ions_per_type):
        species_map.extend([element] * count)

    return species_map


def get_structures_and_forces(
    outcar_path: Union[str, Path],
    poscar_path: Optional[Union[str, Path]] = None,
) -> Tuple[List["Structure"], List[np.ndarray]]:
    """
    Extract all ionic-step structures and forces from a VASP OUTCAR.

    Lattice matrices are updated per ionic step from the OUTCAR itself.
    Species are read natively from POTCAR entries unless *poscar_path* is given.

    Returns
    -------
    (structures, forces)
        structures : list of pymatgen.core.Structure
        forces : list of numpy.ndarray, shape (NIONS, 3), in eV/Å
    """
    from pymatgen.core import Structure

    outcar_path = Path(outcar_path)
    natoms = get_nions(outcar_path)

    if poscar_path:
        from pymatgen.io.vasp import Poscar

        poscar_path = Path(poscar_path)
        if not poscar_path.is_file():
            raise FileNotFoundError(f"POSCAR reference file not found at {poscar_path}")
        species = Poscar.from_file(str(poscar_path)).structure.species
    else:
        species = get_species_and_index_map(outcar_path)

    current_lattice = None
    structures: List[Structure] = []
    forces: List[np.ndarray] = []

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        iterator = iter(f)
        for line in iterator:
            try:
                if "VOLUME and BASIS-vectors are now :" in line:
                    for _ in range(4):
                        next(iterator)
                    lattice_matrix = []
                    for _ in range(3):
                        lattice_matrix.append(
                            [float(x) for x in next(iterator).split()[:3]]
                        )
                    current_lattice = np.array(lattice_matrix)

                if "POSITION" in line and "TOTAL-FORCE" in line:
                    if current_lattice is None:
                        raise ValueError(
                            f"Parsed a POSITION block before finding a lattice matrix "
                            f"in {outcar_path}. The file layout might be corrupted."
                        )

                    next(iterator)  # skip dashed separator
                    coords = np.zeros((natoms, 3))
                    step_forces = np.zeros((natoms, 3))

                    for i in range(natoms):
                        data = next(iterator).split()
                        coords[i] = [float(data[0]), float(data[1]), float(data[2])]
                        step_forces[i] = [
                            float(data[3]),
                            float(data[4]),
                            float(data[5]),
                        ]

                    struct = Structure(
                        lattice=current_lattice,
                        species=species,
                        coords=coords,
                        coords_are_cartesian=True,
                    )
                    structures.append(struct)
                    forces.append(step_forces)

            except StopIteration:
                raise ValueError(
                    f"Premature end of file encountered while parsing {outcar_path}"
                )

    return structures, forces


def get_final_structure_and_forces_from_outcar(
    outcar_path: Union[str, Path],
    poscar_path: Optional[Union[str, Path]] = None,
) -> Tuple["Structure", np.ndarray]:
    """Return only the last structure and forces from an OUTCAR."""
    structures, forces = get_structures_and_forces(outcar_path, poscar_path=poscar_path)
    return structures[-1], forces[-1]


def get_first_structure_and_forces_from_outcar(
    outcar_path: Union[str, Path],
    poscar_path: Optional[Union[str, Path]] = None,
) -> Tuple["Structure", np.ndarray]:
    """Return only the first structure and forces from an OUTCAR."""
    structures, forces = get_structures_and_forces(outcar_path, poscar_path=poscar_path)
    return structures[0], forces[0]


class OutcarParser:
    """
    Lightweight VASP OUTCAR parser (no pymatgen import at construction time).

    Parameters
    ----------
    filename : str or Path
        Path to the VASP OUTCAR file.
    """

    def __init__(self, filename: Union[str, Path]):
        self.filename_path = Path(filename).resolve()
        self.natoms = self.get_natoms()

    def get_natoms(self) -> int:
        self.natoms = get_nions(self.filename_path)
        return self.natoms

    def get_structures_and_forces(
        self, poscar_path: Optional[Union[str, Path]] = None
    ) -> Tuple[List["Structure"], List[np.ndarray]]:
        return get_structures_and_forces(self.filename_path, poscar_path=poscar_path)

    def get_final_structure_and_forces(
        self, poscar_path: Optional[Union[str, Path]] = None
    ) -> Tuple["Structure", np.ndarray]:
        structures, forces = self.get_structures_and_forces(poscar_path=poscar_path)
        return structures[-1], forces[-1]

    def check_convergence(self) -> Dict[str, bool]:
        return check_outcar_convergence(self.filename_path)


# =====================================================================
# Structure-level displacement helpers
# =====================================================================


def _to_structure(obj: Union["Structure", str, Path]) -> "Structure":
    """Coerce a Structure, str, or Path into a pymatgen Structure."""
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Poscar

    if isinstance(obj, (str, Path)):
        return Poscar.from_file(str(obj)).structure
    if isinstance(obj, Structure):
        return obj
    raise TypeError(
        f"Unsupported type: {type(obj)}. Expected pymatgen Structure, str, or Path."
    )


def calc_dR(
    contcar_gs: Union["Structure", str, Path],
    contcar_es: Union["Structure", str, Path],
) -> np.ndarray:
    """
    Compute PBC-safe Cartesian displacement vectors ΔR = R(excited) − R(ground).

    Returns
    -------
    np.ndarray, shape (natoms, 3)  — displacements in Å.
    """
    from pymatgen.util.coord import pbc_shortest_vectors

    struct_gs = _to_structure(contcar_gs)
    struct_es = _to_structure(contcar_es)

    if len(struct_gs) != len(struct_es):
        raise ValueError(
            "Lattice structure mismatch: Input configurations have differing site counts."
        )

    n = len(struct_gs)
    lattice = struct_gs.lattice
    dR = np.vstack(
        [
            pbc_shortest_vectors(
                lattice, struct_gs.frac_coords[i], struct_es.frac_coords[i]
            )
            for i in range(n)
        ]
    ).reshape(n, 3)
    return dR


def calc_delta_Q(struct1: "Structure", struct2: "Structure") -> float:
    """
    Mass-weighted configuration coordinate displacement ΔQ between two structures.

    Returns
    -------
    float  — ΔQ in amu^(1/2) · Å.
    """
    if len(struct1) != len(struct2):
        raise ValueError("Structures must have the same number of atoms.")

    masses = np.array([site.specie.atomic_mass for site in struct1.sites])
    dR = calc_dR(struct1, struct2)
    return calc_delQ(masses, dR)


def get_q_from_structure(
    ground: "Structure",
    excited: "Structure",
    struct: Union["Structure", str, Path],
    tol: float = 1e-4,
    nround: int = 4,
) -> float:
    """
    Project a displaced structure onto the ground→excited configuration coordinate.

    Returns
    -------
    float  — Q in amu^(1/2) · Å.
    """
    from pymatgen.util.coord import pbc_shortest_vectors

    if isinstance(struct, (str, Path)):
        from pymatgen.core import Structure

        tstruct = Structure.from_file(str(struct))
    else:
        tstruct = struct

    if len(ground) != len(excited) or len(ground) != len(tstruct):
        raise ValueError(
            "Lattice structure mismatch: Input geometries have differing site counts."
        )

    masses = np.array([site.specie.atomic_mass for site in ground], dtype=float)
    lattice = ground.lattice

    dr_excited_raw = pbc_shortest_vectors(
        lattice, ground.frac_coords, excited.frac_coords
    )
    dr_excited_raw = np.reshape(dr_excited_raw, (len(ground), 3))

    total_dQ = float(np.sqrt(np.sum(masses * np.sum(dr_excited_raw**2, axis=1))))

    dx_struct = pbc_shortest_vectors(lattice, ground.frac_coords, tstruct.frac_coords)
    dx_struct = np.reshape(dx_struct, (len(ground), 3))

    active_mask = np.abs(dr_excited_raw) > tol
    if not np.any(active_mask):
        return 0.0

    ratios = dx_struct[active_mask] / dr_excited_raw[active_mask]
    rounded_ratios = np.round(ratios, decimals=nround)

    values, counts = np.unique(rounded_ratios, return_counts=True)
    scaling_factor = float(values[np.argmax(counts)])
    return scaling_factor * total_dQ


# =====================================================================
# Force-difference helpers
# =====================================================================


def calc_dF(ground_data: dict, excited_data: dict) -> np.ndarray:
    """
    Compute the force difference dF = F_excited − F_ground at the same geometry.

    Both dicts must have keys ``"structure"`` (pymatgen Structure) and
    ``"forces"`` (numpy array, shape (NIONS, 3)).
    """
    if ground_data["structure"] != excited_data["structure"]:
        raise ValueError(
            "Ground and excited state structures do not match. Cannot calculate dF."
        )
    return excited_data["forces"] - ground_data["forces"]


def prepare_dF_files(
    ground_outcar: Union[str, Path],
    excited_outcar: Union[str, Path],
    ground_poscar: Optional[Union[str, Path]] = None,
    excited_poscar: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Extract dF = F_excited − F_ground from two VASP OUTCAR files.

    Uses the **final** structure/forces from each OUTCAR.
    """
    ground_structure, ground_forces = get_final_structure_and_forces_from_outcar(
        ground_outcar, poscar_path=ground_poscar
    )
    excited_structure, excited_forces = get_final_structure_and_forces_from_outcar(
        excited_outcar, poscar_path=excited_poscar
    )
    ground_data = {"structure": ground_structure, "forces": ground_forces}
    excited_data = {"structure": excited_structure, "forces": excited_forces}
    return calc_dF(ground_data, excited_data)


# =====================================================================
# CCD workflow helpers
# =====================================================================


def generate_ccd_calculations(
    gs_structure: "Structure",
    es_structure: "Structure",
    displacements: List[float],
    output_dir: Union[str, Path],
    ground_template_dir: Union[str, Path],
    excited_template_dir: Union[str, Path],
) -> None:
    """Generate interpolated VASP calculation directories along the CCD path."""
    from defectpl.defectpl import ConfigurationCoordinateDiagram

    ccd = ConfigurationCoordinateDiagram(
        ground_struct=gs_structure, excited_struct=es_structure
    )
    ccd.setup_calculations(
        displacements=displacements,
        output_dir=output_dir,
        ground_input_dir=ground_template_dir,
        excited_input_dir=excited_template_dir,
    )


def analyze_ccd_framework(
    gs_structure: "Structure",
    es_structure: "Structure",
    ground_vaspruns: List[Union[str, Path]],
    excited_vaspruns: List[Union[str, Path]],
    dE: float = 0.0,
    save_plot: Optional[Union[str, Path]] = None,
) -> Tuple[float, float]:
    """Fit harmonic PES wells and return effective phonon frequencies."""
    from defectpl.defectpl import ConfigurationCoordinateDiagram

    ccd = ConfigurationCoordinateDiagram(
        ground_struct=gs_structure, excited_struct=es_structure
    )
    paths_gs = [Path(p) for p in ground_vaspruns]
    paths_es = [Path(p) for p in excited_vaspruns]

    w_g, w_e = ccd.analyze_ccd(
        ground_vaspruns=paths_gs,
        excited_vaspruns=paths_es,
        dE=dE,
        plot=True,
        save_plot=save_plot,
    )
    ccd.estimate_vertical_transitions(ground_omega=w_g, excited_omega=w_e, dE=dE)
    return w_g, w_e


# =====================================================================
# High-level PL calculation runners
# =====================================================================


def run_pl_calc_vasp_displacement_mode(
    band_yaml: Union[str, Path],
    contcar_gs: Union[str, Path],
    contcar_es: Union[str, Path],
    out_dir: Union[str, Path],
    ezpl: float,
    gamma: float,
    plot_all: bool,
    fig_format: str,
) -> None:
    """Run a displacement-mode PL calculation from CONTCAR + band.yaml inputs."""
    try:
        from pymatgen.core import Structure

        from defectpl.defectpl import Photoluminescence
        from defectpl.phonon import read_band_yaml

        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        struct_gs = Structure.from_file(contcar_gs)
        struct_es = Structure.from_file(contcar_es)
        dR = calc_dR(struct_gs, struct_es)

        pl_engine = Photoluminescence(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=dR,
            EZPL=ezpl,
            gamma=gamma,
            max_energy=5.0,
            sigma=6e-3,
        )

        if plot_all:
            pl_engine.generate_plots(out_dir=out_dir, fig_format=fig_format)

    except Exception as exc:
        raise RuntimeError(f"Calculation pipeline failure encountered: {exc}")


def run_pl_calc_vasp_force_mode(
    band_yaml: Union[str, Path],
    outcar_gs: Union[str, Path],
    outcar_es: Union[str, Path],
    out_dir: Union[str, Path],
    ezpl: float,
    gamma: float,
    plot_all: bool,
    fig_format: str,
) -> None:
    """Run a force-mode PL calculation from OUTCAR + band.yaml inputs."""
    try:
        from defectpl.defectpl import Photoluminescence
        from defectpl.phonon import read_band_yaml

        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        dF = prepare_dF_files(outcar_gs, outcar_es)

        pl_engine = Photoluminescence(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dF=dF,
            EZPL=ezpl,
            gamma=gamma,
            max_energy=5.0,
            sigma=6e-3,
        )

        if plot_all:
            pl_engine.generate_plots(out_dir=out_dir, fig_format=fig_format)

    except Exception as exc:
        raise RuntimeError(f"Calculation pipeline failure encountered: {exc}")


def run_dynamic_yaml_comparison(
    band_yaml_files: List[Union[str, Path]],
    gs_structure: "Structure",
    es_structure: "Structure",
    out_dir: Union[str, Path],
    ezpl: float,
    gamma: float,
    xmin: float,
    xmax: float,
    file_name: str,
) -> Path:
    """Build and plot comparative PL intensities for multiple band.yaml files."""
    import matplotlib.pyplot as plt

    from defectpl.defectpl import Photoluminescence
    from defectpl.phonon import read_band_yaml

    dR = calc_dR(gs_structure, es_structure)
    all_pl_runs = []

    for b_yaml in band_yaml_files:
        freqs, evecs, masses = read_band_yaml(b_yaml)
        pl_run = Photoluminescence(
            frequencies=freqs,
            eigenvectors=evecs,
            masses=masses,
            dR=dR,
            EZPL=ezpl,
            gamma=gamma,
            max_energy=5.0,
            sigma=6e-3,
        )
        all_pl_runs.append(pl_run)

    plt.figure(figsize=(4, 4))
    for pl_run in all_pl_runs:
        plt.plot(pl_run.intensity.__abs__(), "k", lw=1.2)

    plt.ylabel(r"$I(\hbar\omega)$")
    plt.xlabel(r"Photon energy (eV)")
    plt.xlim(xmin, xmax)

    x_values, _ = plt.xticks()
    resolution = all_pl_runs[0].resolution if all_pl_runs else 1.0
    labels = [float(x) / resolution for x in x_values]
    plt.xticks(x_values, labels)

    out_path = Path(out_dir) / file_name
    fmt = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", format=fmt)
    plt.close()

    return out_path


def run_kohn_sham_analysis(
    eigenval_path: Union[str, Path],
    vbm: float,
    cbm: float,
    espan: float = 1.0,
    k_idx: int = 0,
    output_img: Union[str, Path] = "ks_plot.png",
    output_json: Optional[Union[str, Path]] = None,
) -> None:
    """Parse EIGENVAL, resolve degeneracies, and plot KS electronic levels."""
    from defectpl.ks_analysis import (
        extract_ksplot_data,
        plot_spin_resolved_levels,
    )

    raw_data = read_eigenval_file(eigenval_path, k_idx=k_idx)
    ks_model = extract_ksplot_data(raw_data, vbm=vbm, cbm=cbm, espan=espan)

    if output_json:
        Path(output_json).write_text(ks_model.to_json(), encoding="utf-8")

    plot_spin_resolved_levels(ks_model, output_filename=output_img)


# =====================================================================
# VaspReader — implements PhononReader and ElectronicReader protocols
# =====================================================================


class VaspReader:
    """
    VASP-specific implementation of the :class:`~defectpl.io.base.PhononReader`
    and :class:`~defectpl.io.base.ElectronicReader` protocols.

    All DFT-code-specific logic lives here; the physics layer only sees
    the generic :class:`~defectpl.core.structures.PhononData` /
    :class:`~defectpl.core.structures.EigenvalData` containers.
    """

    # ---- PhononReader ---------------------------------------------------

    def read_band_yaml(self, path: str) -> "PhononData":  # noqa: F821
        from defectpl.core.structures import PhononData
        from defectpl.phonon import read_band_yaml as _read

        freqs, evecs, masses = _read(path)
        return PhononData(
            frequencies=freqs,
            eigenvectors=evecs,
            masses=masses,
            natoms=len(masses),
            nmodes=len(freqs),
            meta={"source_file": str(path), "code": "vasp"},
        )

    def read_force_constants(self, path: str) -> np.ndarray:
        try:
            from phonopy.file_IO import parse_FORCE_CONSTANTS
        except ImportError as exc:
            raise ImportError(
                "phonopy is required to read FORCE_CONSTANTS. "
                "Install with: pip install phonopy"
            ) from exc
        return parse_FORCE_CONSTANTS(filename=str(path))

    def read_forces(self, path: str, state: str = "gs") -> np.ndarray:
        _, forces = get_final_structure_and_forces_from_outcar(path)
        return forces

    # ---- ElectronicReader -----------------------------------------------

    def read_eigenvalues(self, path: str, k_idx: int = 0) -> "EigenvalData":  # noqa: F821
        from defectpl.core.structures import EigenvalData

        data = read_eigenval_file(path, k_idx=k_idx)
        return EigenvalData(**data)

    def read_structures(self, path: str) -> Tuple[list, list]:
        return get_structures_and_forces(path)
