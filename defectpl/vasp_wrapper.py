# -*- coding: utf-8 -*-
"""
Automation wrapper utilities handling structural interpolations, job generation, 
and output database array parsing for VASP calculations.
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np


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

    ccd = ConfigurationCoordinateDiagram(ground_struct=gs_structure, excited_struct=es_structure)
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

    ccd = ConfigurationCoordinateDiagram(ground_struct=gs_structure, excited_struct=es_structure)
    
    paths_gs = [Path(p) for p in ground_vaspruns]
    paths_es = [Path(p) for p in excited_vaspruns]

    # Fit quadratic potential curves to find effective generalized phonon frequencies
    w_g, w_e = ccd.analyze_ccd(
        ground_vaspruns=paths_gs,
        excited_vaspruns=paths_es,
        dE=dE,
        plot=True,
        save_plot=save_plot
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
            sigma=6e-3
        )
        all_pl_runs.append(pl_run)

    plt.figure(figsize=(4, 4))
    for pl_run in all_pl_runs:
        plt.plot(pl_run.I.__abs__(), "k", lw=1.2)
        
    plt.ylabel(r"$I(\hbar\omega)$")
    plt.xlabel(r"Photon energy (eV)")
    plt.xlim(xmin, xmax)
    
    x_values, _ = plt.xticks()
    resolution = all_pl_runs[0].resolution if hasattr(all_pl_runs[0], "resolution") else 1.0
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