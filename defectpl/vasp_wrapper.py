# -*- coding: utf-8 -*-
"""
Wrapper functions to interface with VASP output files and extract necessary data for defect photoluminescence calculations, 
including structure parsing, force extraction, and configuration coordinate analysis. This module serves as a bridge between 
raw VASP outputs and the higher-level photoluminescence modeling workflows defined in defectpl.defectpl.
"""

from pathlib import Path
from typing import List, Tuple, Union

import click
import numpy as np

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.util.coord import pbc_shortest_vectors

from defectpl.phonon import read_band_yaml
from defectpl.vasp import (
    get_final_structure_and_forces_from_outcar,
    get_first_structure_and_forces_from_outcar,
)
from defectpl.utils import calc_delQ


def generate_ccd_calculations(
    gs_structure,
    es_structure,
    displacements: List[float],
    output_dir: Union[str, Path],
    ground_template_dir: Union[str, Path],
    excited_template_dir: Union[str, Path],
) -> None:
    """
    Generate linear interpolation structure configuration spaces for automated VASP execution parameters.

    This splits out structure interpolation and replicates input profiles (INCAR, POTCAR, KPOINTS)
    from template directories for each step.
    """
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
    gs_structure,
    es_structure,
    ground_vaspruns: List[Union[str, Path]],
    excited_vaspruns: List[Union[str, Path]],
    dE: float = 0.0,
    save_plot: Union[str, Path] = None,
) -> Tuple[float, float]:
    """
    Fit calculated Potential Energy Surfaces data arrays, extract well parameters,
    and report vertical transition metrics.
    """
    from defectpl.defectpl import ConfigurationCoordinateDiagram

    ccd = ConfigurationCoordinateDiagram(
        ground_struct=gs_structure, excited_struct=es_structure
    )

    paths_gs = [Path(p) for p in ground_vaspruns]
    paths_es = [Path(p) for p in excited_vaspruns]

    # Fit quadratic potential curves to find effective generalized phonon frequencies
    w_g, w_e = ccd.analyze_ccd(
        ground_vaspruns=paths_gs,
        excited_vaspruns=paths_es,
        dE=dE,
        plot=True,
        save_plot=save_plot,
    )

    # Print vertical absorption, emission, and Jahn-Teller parameters to the standard output log
    ccd.estimate_vertical_transitions(ground_omega=w_g, excited_omega=w_e, dE=dE)

    return w_g, w_e


def run_dynamic_yaml_comparison(
    band_yaml_files: List[Union[str, Path]],
    gs_structure,
    es_structure,
    out_dir: Union[str, Path],
    ezpl: float,
    gamma: float,
    xmin: float,
    xmax: float,
    file_name: str,
) -> Path:
    """
    Dynamically compile, build, and plot comparative intensities for lists of
    phonopy band configuration inputs.
    """
    import matplotlib.pyplot as plt
    from defectpl.utils import read_band_yaml, calc_dR
    from defectpl.defectpl import Photoluminescence

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
        plt.plot(pl_run.I.__abs__(), "k", lw=1.2)

    plt.ylabel(r"$I(\hbar\omega)$")
    plt.xlabel(r"Photon energy (eV)")
    plt.xlim(xmin, xmax)

    x_values, _ = plt.xticks()
    resolution = (
        all_pl_runs[0].resolution if hasattr(all_pl_runs[0], "resolution") else 1.0
    )
    labels = [float(x) / resolution for x in x_values]
    plt.xticks(x_values, labels)

    out_path = Path(out_dir) / file_name
    form = file_name.split(".")[-1]
    fmt = form.lower() if form else "pdf"

    plt.savefig(out_path, dpi=300, bbox_inches="tight", format=fmt)
    plt.close()

    return out_path


def run_kohn_sham_analysis(
    eigenval_path: str,
    vbm: float,
    cbm: float,
    espan: float = 1.0,
    k_idx: int = 0,
    output_img: str = "ks_plot.png",
    output_json: str = None,
) -> None:
    """
    Orchestrates the raw EIGENVAL parsing, structural energy level truncation,
    degeneracy mapping, serialization, and plotting workflows.

    Parameters
    ----------
    eigenval_path : str
        Path to the VASP EIGENVAL or EIGENVAL.gz file.
    vbm : float
        Energy of the Valence Band Maximum (eV).
    cbm : float
        Energy of the Conduction Band Minimum (eV).
    espan : float, default 1.0
        Energy buffer span tracking above/below band edges.
    k_idx : int, default 0
        K-point array slice index to analyze.
    output_img : str, default "ks_plot.png"
        Destination file path for the final image.
    output_json : str, optional
        If provided, dumps the MSONable dataclass JSON stream to this path.
    """
    from defectpl.ks_analysis import (
        read_eigenval_file,
        extract_ksplot_data,
        plot_spin_resolved_levels,
    )
    from pathlib import Path

    # 1. Parse raw VASP eigenvalue blocks
    raw_data = read_eigenval_file(eigenval_path, k_idx=k_idx)

    # 2. Extract degeneracy groupings and plot coordinates
    ks_model = extract_ksplot_data(raw_data, vbm=vbm, cbm=cbm, espan=espan)

    # 3. Handle data serialization to disk if requested
    if output_json:
        p = Path(output_json)
        p.write_text(ks_model.to_json(), encoding="utf-8")

    # 4. Generate visual levels layout matrix
    plot_spin_resolved_levels(ks_model, output_filename=output_img)


def run_pl_calc_vasp_displacement_mode(
    band_yaml,
    contcar_gs,
    contcar_es,
    out_dir,
    ezpl,
    gamma,
    plot_all,
    fig_format,
):
    try:
        from defectpl.defectpl import Photoluminescence

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


def calc_dF(ground_data, excited_data):
    """
    Given the forces at a structure, in ground and excited states, calculate the force difference vector dF, where
    dF = F_excited - F_ground at the same structure.

    Parameters:
    -----------
    ground_data : dict
        A dictionary containing structure and forces for the ground state.
        Expected keys: 'structure' (pymatgen Structure) and 'forces' (numpy array).
        Note: Take the last structure and forces from the ground state VASP run, which should correspond to the same structure
        as the vertical excitation point.
    excited_data : dict
        A dictionary containing structure and forces for the excited state.
        Expected keys: 'structure' (pymatgen Structure) and 'forces' (numpy array).

        Note: Take the first structure and forces from the excited state VASP run, which should correspond to the same structure
        as the ground state structure at the vertical excitation point.

    Returns:
    --------
    dF : numpy array
        The force difference vector, calculated as F_excited - F_ground.
    """
    # Check whether the structures are the same
    if ground_data["structure"] != excited_data["structure"]:
        raise ValueError(
            "Ground and excited state structures do not match. Cannot calculate dF."
        )

    # Extract forces
    F_ground = ground_data["forces"]
    F_excited = excited_data["forces"]
    # Calculate force difference
    dF = F_excited - F_ground
    return dF


def prepare_dF_files(
    ground_outcar, excited_outcar, ground_poscar=None, excited_poscar=None
):
    """
    Prepare the force difference vector dF by extracting forces from VASP OUTCAR files for both ground and excited states.

    Parameters:
    -----------
    ground_outcar : str
        Path to the OUTCAR file from the ground state VASP calculation.
    excited_outcar : str
        Path to the OUTCAR file from the excited state VASP calculation.
    ground_poscar : str, optional
        Path to the POSCAR file for the ground state structure. Required if species information is needed.
    excited_poscar : str, optional
        Path to the POSCAR file for the excited state structure. Required if species information is needed.

    Returns:
    --------
    dF : numpy array
        The calculated force difference vector dF.
    """
    # Extract structures and forces from OUTCAR files
    ground_structure, ground_forces = get_final_structure_and_forces_from_outcar(
        ground_outcar, poscar_path=ground_poscar
    )
    excited_structure, excited_forces = get_final_structure_and_forces_from_outcar(
        excited_outcar, poscar_path=excited_poscar
    )

    # Prepare data dictionaries for dF calculation
    ground_data = {"structure": ground_structure, "forces": ground_forces}
    excited_data = {"structure": excited_structure, "forces": excited_forces}

    # Calculate dF
    dF = calc_dF(ground_data, excited_data)

    return dF


def run_pl_calc_vasp_force_mode(
    band_yaml,
    outcar_gs,
    outcar_es,
    out_dir,
    ezpl,
    gamma,
    plot_all,
    fig_format,
):
    try:
        from defectpl.defectpl import Photoluminescence

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


def _to_structure(obj: Union[Structure, str, Path]) -> Structure:
    """Helper to convert a Structure, str, or Path into a pymatgen Structure."""
    if isinstance(obj, (str, Path)):
        # Support both regular POSCAR/CONTCAR syntax files
        return Poscar.from_file(str(obj)).structure
    elif isinstance(obj, Structure):
        return obj
    else:
        raise TypeError(
            f"Unsupported type: {type(obj)}. Expected pymatgen Structure, str, or Path."
        )


def calc_dR(
    constcar_gs: Union[Structure, str, Path], contcar_es: Union[Structure, str, Path]
) -> np.ndarray:
    """
    Calculates the shortest periodic boundary condition (PBC) safe displacement
    vectors between the ground state and excited state structures.

    Parameters:
    ===========
    constcar_gs : Structure, str, or Path
        Ground state equilibrium configuration (Structure object or file path).
    contcar_es : Structure, str, or Path
        Excited state equilibrium configuration (Structure object or file path).

    Returns:
    =================
    dR : np.ndarray
        2D array of shape (n_atoms, 3) detailing Cartesian displacements
        (Excited - Ground) in Å.
    """
    # Normalize both inputs to pymatgen Structure objects
    struct_gs = _to_structure(constcar_gs)
    struct_es = _to_structure(contcar_es)

    if len(struct_gs) != len(struct_es):
        raise ValueError(
            "Lattice structure mismatch: Input configurations have differing site counts."
        )

    length = len(struct_gs)
    lattice = struct_gs.lattice
    dR = np.vstack(
        [
            pbc_shortest_vectors(
                lattice, struct_gs.frac_coords[i], struct_es.frac_coords[i]
            )
            for i in range(length)
        ]
    ).reshape(length, 3)
    return dR


def calc_delta_Q(struct1: Structure, struct2: Structure) -> float:
    """
    Calculate the mass-weighted coordinate displacement delta Q between structures.

    Parameters
    ----------
    struct1 : Structure
        The initial reference equilibrium configuration pymatgen Structure object.
    struct2 : Structure
        The final reference equilibrium configuration pymatgen Structure object.

    Returns
    -------
    float
        The total mass-weighted structural displacement delta Q in amu^(1/2) * Å.

    Raises
    ------
    ValueError
        If structures possess unequal quantities of internal atomic site objects.
    """
    if len(struct1) != len(struct2):
        raise ValueError("Structures must have the same number of atoms.")

    masses = np.array([site.specie.atomic_mass for site in struct1.sites])

    dR = calc_dR(struct1, struct2)
    delta_Q = calc_delQ(masses, dR)

    return delta_Q


def get_q_from_structure(
    ground: Structure,
    excited: Structure,
    struct: Union[Structure, str, Path],
    tol: float = 1e-4,
    nround: int = 4,
) -> float:
    """
    Calculates the mass-weighted configuration coordinate (Q) value for a given
    displaced configuration relative to the equilibrium ground-state structure.

    Parameters:
    ===========
    ground : Structure
        Equilibrium geometry structure for the electronic ground state.
    excited : Structure
        Equilibrium geometry structure for the electronic excited state.
    struct : Structure, str, or Path
        The intermediate or displaced structure (or path to its file) to evaluate.
    tol : float, default 1e-4
        Distance threshold filter cutoff (Å) to strip stationary lattice atoms
        out of the interpolation projection tracking step.
    nround : int, default 4
        Decimal rounding precision used when clustering atomic displacement ratios.

    Returns:
    ========
    Q : float
        Calculated configuration coordinate position step in units of amu^(1/2) * Å.
    """
    if isinstance(struct, (str, Path)):
        tstruct = Structure.from_file(str(struct))
    else:
        tstruct = struct

    if len(ground) != len(excited) or len(ground) != len(tstruct):
        raise ValueError(
            "Lattice structure mismatch: Input geometries have differing site counts."
        )

    masses = np.array([site.specie.atomic_mass for site in ground], dtype=float)
    lattice = ground.lattice

    # Vectorized PBC shortest vector layout matching calc_delta_Q
    dr_excited_raw = pbc_shortest_vectors(
        lattice, ground.frac_coords, excited.frac_coords
    )
    dr_excited_raw = np.reshape(dr_excited_raw, (len(ground), 3))

    total_dQ = float(np.sqrt(np.sum(masses * np.sum(dr_excited_raw**2, axis=1))))

    dx_excited = dr_excited_raw
    dx_struct = pbc_shortest_vectors(lattice, ground.frac_coords, tstruct.frac_coords)
    dx_struct = np.reshape(dx_struct, (len(ground), 3))

    active_mask = np.abs(dx_excited) > tol

    if not np.any(active_mask):
        return 0.0

    ratios = dx_struct[active_mask] / dx_excited[active_mask]
    rounded_ratios = np.round(ratios, decimals=nround)

    values, counts = np.unique(rounded_ratios, return_counts=True)
    scaling_factor = float(values[np.argmax(counts)])

    return scaling_factor * total_dQ
