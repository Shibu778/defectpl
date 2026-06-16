# -*- coding: utf-8 -*-
"""
Module for parsing, processing, and plotting Kohn-Sham eigenvalues from VASP calculations.

This module provides data models, I/O functions, and visualization tools to analyze
defect states and electronic transitions near the semiconductor bandgap.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from monty.json import MSONable

# Cross-module package imports supporting decoupled VASP I/O pipelines
from defectpl.vasp import get_spin_multiplicity, read_eigenval_file


class KohnShamPlotData(MSONable):
    """
    An MSONable data container storing processed spin-polarized eigenvalues.

    Stores energy data, occupations, degeneracy properties, and lateral layout
    coordinates required for plotting electronic levels near the band edges.

    Parameters
    ----------
    up : list of list of float
        Truncated spin-up states represented as [energy, occupancy] pairs.
    down : list of list of float
        Truncated spin-down states represented as [energy, occupancy] pairs.
    up_idx : list of int
        Original band indices for the truncated spin-up channels.
    down_idx : list of int
        Original band indices for the truncated spin-down channels.
    up_energies : list of float
        Extracted energies for the truncated spin-up states.
    up_occupations : list of float
        Extracted occupation values for the truncated spin-up states.
    down_energies : list of float
        Extracted energies for the truncated spin-down states.
    down_occupations : list of float
        Extracted occupation values for the truncated spin-down states.
    degenerate_up : list of list of int
        Sublists grouping indices of degenerate spin-up eigenvalues.
    degenerate_down : list of list of int
        Sublists grouping indices of degenerate spin-down eigenvalues.
    max_div_up : int
        Maximum degree of degeneracy found in any spin-up cluster.
    max_div_down : int
        Maximum degree of degeneracy found in any spin-down cluster.
    xvalues_up : list of float
        Calculated negative horizontal positions for plotting spin-up states.
    xvalues_down : list of float
        Calculated positive horizontal positions for plotting spin-down states.
    occupied_up : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of occupied up states.
    unoccupied_up : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of empty up states.
    occupied_down : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of occupied down states.
    unoccupied_down : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of empty down states.
    vbm : float
        Energy of the Valence Band Maximum (eV).
    cbm : float
        Energy of the Conduction Band Minimum (eV).
    emin : float
        Minimum energy boundary displayed on the plot canvas (eV).
    emax : float
        Maximum energy boundary displayed on the plot canvas (eV).
    espan : float
        Energy buffer width padding the VBM and CBM regions (eV).
    sep : float
        Lateral separation distance separating adjacent degenerate markers.
    lim : float
        Maximum horizontal layout boundary constraint.
    w : float
        Calculated horizontal step dimension mapping out state widths.
    meta_info : dict, optional
        Metadata payload covering k-point indices, spins, and electron counts.
    """

    def __init__(
        self,
        up: List[List[float]],
        down: List[List[float]],
        up_idx: List[int],
        down_idx: List[int],
        up_energies: List[float],
        up_occupations: List[float],
        down_energies: List[float],
        down_occupations: List[float],
        degenerate_up: List[List[int]],
        degenerate_down: List[List[int]],
        max_div_up: int,
        max_div_down: int,
        xvalues_up: List[float],
        xvalues_down: List[float],
        occupied_up: Dict[str, List[float]],
        unoccupied_up: Dict[str, List[float]],
        occupied_down: Dict[str, List[float]],
        unoccupied_down: Dict[str, List[float]],
        vbm: float,
        cbm: float,
        emin: float,
        emax: float,
        espan: float,
        sep: float,
        lim: float,
        w: float,
        meta_info: Optional[Dict[str, Any]] = None,
    ):
        self.up = up
        self.down = down
        self.up_idx = up_idx
        self.down_idx = down_idx
        self.up_energies = up_energies
        self.up_occupations = up_occupations
        self.down_energies = down_energies
        self.down_occupations = down_occupations
        self.degenerate_up = degenerate_up
        self.degenerate_down = degenerate_down
        self.max_div_up = max_div_up
        self.max_div_down = max_div_down
        self.xvalues_up = xvalues_up
        self.xvalues_down = xvalues_down
        self.occupied_up = occupied_up
        self.unoccupied_up = unoccupied_up
        self.occupied_down = occupied_down
        self.unoccupied_down = unoccupied_down
        self.vbm = vbm
        self.cbm = cbm
        self.emin = emin
        self.emax = emax
        self.espan = espan
        self.sep = sep
        self.lim = lim
        self.w = w
        self.meta_info = meta_info or {}

    def as_dict(self) -> Dict[str, Any]:
        """
        Serialize the object instance properties into a JSON-compatible dictionary.

        Returns
        -------
        dict
            A key-value serialization dictionary layout mapping object states.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "up": self.up,
            "down": self.down,
            "up_idx": self.up_idx,
            "down_idx": self.down_idx,
            "up_energies": self.up_energies,
            "up_occupations": self.up_occupations,
            "down_energies": self.down_energies,
            "down_occupations": self.down_occupations,
            "degenerate_up": self.degenerate_up,
            "degenerate_down": self.degenerate_down,
            "max_div_up": self.max_div_up,
            "max_div_down": self.max_div_down,
            "xvalues_up": self.xvalues_up,
            "xvalues_down": self.xvalues_down,
            "occupied_up": self.occupied_up,
            "unoccupied_up": self.unoccupied_up,
            "occupied_down": self.occupied_down,
            "unoccupied_down": self.unoccupied_down,
            "vbm": self.vbm,
            "cbm": self.cbm,
            "emin": self.emin,
            "emax": self.emax,
            "espan": self.espan,
            "sep": self.sep,
            "lim": self.lim,
            "w": self.w,
            "meta_info": self.meta_info,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KohnShamPlotData":
        """
        Reconstruct a class instance from an unpacked serialization dictionary.

        Parameters
        ----------
        d : dict
            The dictionary generated via `.as_dict()` or loaded from JSON files.

        Returns
        -------
        KohnShamPlotData
            An instantiated, data-populated class object model.
        """
        return cls(
            up=d["up"],
            down=d["down"],
            up_idx=d["up_idx"],
            down_idx=d["down_idx"],
            up_energies=d["up_energies"],
            up_occupations=d["up_occupations"],
            down_energies=d["down_energies"],
            down_occupations=d["down_occupations"],
            degenerate_up=d["degenerate_up"],
            degenerate_down=d["degenerate_down"],
            max_div_up=d["max_div_up"],
            max_div_down=d["max_div_down"],
            xvalues_up=d["xvalues_up"],
            xvalues_down=d["xvalues_down"],
            occupied_up=d["occupied_up"],
            unoccupied_up=d["unoccupied_up"],
            occupied_down=d["occupied_down"],
            unoccupied_down=d["unoccupied_down"],
            vbm=d["vbm"],
            cbm=d["cbm"],
            emin=d["emin"],
            emax=d["emax"],
            espan=d["espan"],
            sep=d["sep"],
            lim=d["lim"],
            w=d["w"],
            meta_info=d.get("meta_info", {}),
        )


def get_homo_lumo_idx(eigenval: List[List[float]], thr: float = 0.6) -> Tuple[int, int]:
    """
    Locate the Highest Occupied and Lowest Unoccupied Molecular Orbital boundaries.

    Parameters
    ----------
    eigenval : list of list of float
        A list of [energy, occupancy] state arrays targeting a single spin channel.
    thr : float, default 0.6
        The occupancy threshold demarcating occupied from empty electronic levels.

    Returns
    -------
    homo_idx : int
        The index coordinate marking the HOMO band horizon.
    lumo_idx : int
        The index coordinate marking the LUMO band horizon.
    """
    ev_array = np.array(eigenval)
    occupations = ev_array[:, 1]
    if len(set(occupations)) > 2:
        warnings.warn(
            f"Fractional occupancies detected inside dataset. Threshold: {thr} applied."
        )

    homo_idx = max(i for i, occ in enumerate(occupations) if occ > thr)
    lumo_idx = min(i for i, occ in enumerate(occupations) if occ < thr)

    return homo_idx, lumo_idx


def truncate_eigenval(
    eigenval: List[List[float]], emin: float, emax: float
) -> Tuple[List[List[float]], List[int]]:
    """
    Truncate an eigenvalues array to a specified energy range.

    Parameters
    ----------
    eigenval : list of list of float
        A sequence of electronic [energy, occupancy] data pairs.
    emin : float
        The minimum cut-off threshold energy (eV).
    emax : float
        The maximum cut-off threshold energy (eV).

    Returns
    -------
    trunc_eig : list of list of float
        Filtered array entries falling within the specified energetic window.
    index : list of int
        The matching original index coordinates tracked prior to clipping.
    """
    index = [i for i, (e, o) in enumerate(eigenval) if emin <= e <= emax]
    trunc_eig = [eigenval[i] for i in index]
    return trunc_eig, index


def find_degenerate_eigenvalues(
    eigenval: List[List[float]], tol: float = 1e-3
) -> List[List[int]]:
    """
    Group eigenvalues into clusters based on an energy tolerance threshold.

    Parameters
    ----------
    eigenval : list of list of float
        A sequence of electronic [energy, occupancy] data pairs.
    tol : float, default 1e-3
        The absolute energetic proximity tolerance window defining degeneracies (eV).

    Returns
    -------
    list of list of int
        A grouped index hierarchy list tracking degenerate energy clusters.
    """
    degenerate_groups = []
    considered = []
    for i in range(len(eigenval)):
        if i in considered:
            continue
        group = [i]
        considered.append(i)
        for j in range(i + 1, len(eigenval)):
            if abs(eigenval[i][0] - eigenval[j][0]) < tol:
                group.append(j)
                considered.append(j)
        degenerate_groups.append(group)
    return degenerate_groups


def split_energy_occupation(
    eigenval: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """
    Separate paired state entries into distinct energy and occupancy vectors.

    Parameters
    ----------
    eigenval : list of list of float
        A sequence of electronic [energy, occupancy] data pairs.

    Returns
    -------
    energies : list of float
        The extracted isolated energy values.
    occupations : list of float
        The extracted isolated state occupation weights.
    """
    energies = [e for e, o in eigenval]
    occupations = [o for e, o in eigenval]
    return energies, occupations


def xpos_evaluation(
    npoint: int, max_div: int, sep: float = 0.1, lim: float = 10.0
) -> List[float]:
    """
    Calculate lateral positions to resolve degenerate levels along the X-axis.

    Parameters
    ----------
    npoint : int
        The total number of degenerate points nested inside the local focus cluster.
    max_div : int
        The highest overall degeneracy factor matching the spin sequence globally.
    sep : float, default 0.1
        The separation width spacing adjacent markers.
    lim : float, default 10.0
        The spatial canvas layout boundary limit.

    Returns
    -------
    list of float
        A sequence of calculated horizontal plotting offsets.
    """
    if npoint == 0:
        return [0.0]
    midpos = 0.5 * lim
    w = (lim - sep) / max_div - sep

    if npoint == 1:
        return [midpos]
    elif npoint % 2 == 1:
        xpos = [midpos]
        xpos += [midpos + i * w / 2.0 + i * sep / 2.0 for i in range(2, npoint, 2)]
        xpos += [midpos - i * w / 2.0 - i * sep / 2.0 for i in range(2, npoint, 2)]
        return xpos
    else:
        xpos = [midpos + i * w / 2.0 + i * sep / 2.0 for i in range(1, npoint, 2)]
        xpos += [midpos - i * w / 2.0 - i * sep / 2.0 for i in range(1, npoint, 2)]
        return xpos


def get_x_values(
    deg_group: List[List[int]], max_div: int, sep: float = 0.1, lim: float = 10.0
) -> List[float]:
    """
    Aggregate layout offsets sequentially across a cluster array sequence.

    Parameters
    ----------
    deg_group : list of list of int
        A list of sublists containing indices of degenerate eigenvalues.
    max_div : int
        The highest overall degeneracy factor matching the spin sequence globally.
    sep : float, default 0.1
        The separation width spacing adjacent markers.
    lim : float, default 10.0
        The spatial canvas layout boundary limit.

    Returns
    -------
    list of float
        The complete aggregated list of calculated horizontal coordinates.
    """
    xvalues = []
    for group in deg_group:
        npoint = len(group)
        xvalues += xpos_evaluation(npoint, max_div, sep, lim)
    return xvalues


def get_occupied_unoccupied_split(
    occupations: List[float],
    xvalues: List[float],
    energies: List[float],
    threshold: float = 0.6,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Categorize states into occupied and unoccupied coordinate dictionaries.

    Parameters
    ----------
    occupations : list of float
        State occupancy numbers mapped matching target lists.
    xvalues : list of float
        The resolved horizontal positioning variables layout maps.
    energies : list of float
        The isolated electronic band energies.
    threshold : float, default 0.6
        The threshold value separating occupied and empty states.

    Returns
    -------
    occupied : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of filled levels.
    unoccupied : dict of (str, list of float)
        X-positions ('xvalues') and energies ('energies') of empty levels.
    """
    occupied = {"xvalues": [], "energies": []}
    unoccupied = {"xvalues": [], "energies": []}
    for occ, x, energy in zip(occupations, xvalues, energies):
        if occ > threshold:
            occupied["xvalues"].append(x)
            occupied["energies"].append(energy)
        else:
            unoccupied["xvalues"].append(x)
            unoccupied["energies"].append(energy)
    return occupied, unoccupied


def extract_ksplot_data(
    eigenval_data: Dict[str, Any],
    vbm: float,
    cbm: float,
    espan: float = 1.0,
    sep: float = 0.1,
    lim: float = 10.0,
) -> KohnShamPlotData:
    """
    Process raw k-point eigenvalue metrics to build a complete plotting data model.

    Extracts windows, maps degeneracy spacing matrices, updates offsets,
    and returns a serialized data model container.

    Parameters
    ----------
    eigenval_data : dict
        Raw unpacked dictionary generated via `read_eigenval_file`.
    vbm : float
        Energy of the Valence Band Maximum (eV).
    cbm : float
        Energy of the Conduction Band Minimum (eV).
    espan : float, default 1.0
        Energy buffer width padding the VBM and CBM boundary fields (eV).
    sep : float, default 0.1
        Lateral separation parameter adjusting near-degenerate points.
    lim : float, default 10.0
        The maximum lateral geometric width limit.

    Returns
    -------
    KohnShamPlotData
        A data container populated with processed levels and spatial coordinates.
    """
    emin = vbm - espan
    emax = cbm + espan

    up_trunc, up_idx = truncate_eigenval(eigenval_data["up"], emin, emax)
    down_trunc, down_idx = truncate_eigenval(eigenval_data["down"], emin, emax)

    up_energies, up_occupations = split_energy_occupation(up_trunc)
    down_energies, down_occupations = split_energy_occupation(down_trunc)

    degenerate_up = find_degenerate_eigenvalues(up_trunc)
    degenerate_down = find_degenerate_eigenvalues(down_trunc)

    max_div_up = max(len(g) for g in degenerate_up) if degenerate_up else 1
    max_div_down = max(len(g) for g in degenerate_down) if degenerate_down else 1

    xvalues_up = [-x for x in get_x_values(degenerate_up, max_div_up, sep, lim)]
    xvalues_down = get_x_values(degenerate_down, max_div_down, sep, lim)

    occupied_up, unoccupied_up = get_occupied_unoccupied_split(
        up_occupations, xvalues_up, up_energies
    )
    occupied_down, unoccupied_down = get_occupied_unoccupied_split(
        down_occupations, xvalues_down, down_energies
    )

    min_div = min(max_div_up, max_div_down)
    w = (lim - sep) / min_div - sep if min_div > 0 else (lim - sep)

    meta_info = {
        "selected_kpoint": eigenval_data.get("selected_kpoint"),
        "spin_multiplicity": eigenval_data.get("spin_multiplicity"),
        "nelect": eigenval_data.get("nelect"),
    }

    return KohnShamPlotData(
        up=up_trunc,
        down=down_trunc,
        up_idx=up_idx,
        down_idx=down_idx,
        up_energies=up_energies,
        up_occupations=up_occupations,
        down_energies=down_energies,
        down_occupations=down_occupations,
        degenerate_up=degenerate_up,
        degenerate_down=degenerate_down,
        max_div_up=max_div_up,
        max_div_down=max_div_down,
        xvalues_up=xvalues_up,
        xvalues_down=xvalues_down,
        occupied_up=occupied_up,
        unoccupied_up=unoccupied_up,
        occupied_down=occupied_down,
        unoccupied_down=unoccupied_down,
        vbm=vbm,
        cbm=cbm,
        emin=emin,
        emax=emax,
        espan=espan,
        sep=sep,
        lim=lim,
        w=w,
        meta_info=meta_info,
    )


def plot_spin_resolved_levels(
    data: KohnShamPlotData,
    output_filename: Union[str, Path] = "ks_plot.png",
    style_file: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot Kohn-Sham energy levels with separate spin-up and spin-down panels.

    Renders energy levels alongside shaded regions for the valence and conduction
    bands, and marks state occupancy. Supports native Matplotlib Axes injection.

    Parameters
    ----------
    data : KohnShamPlotData
        The processed data container containing the state properties and layout coordinates.
    output_filename : str or pathlib.Path, default "ks_plot.png"
        The path and file name where the final plot image will be saved if ax is None.
    style_file : str, optional
        An optional path to a matplotlib `.mplstyle` configuration file.
    ax : matplotlib.axes.Axes, optional
        An existing Matplotlib Axes object to paint the plot on. If None, a new
        standalone figure is instantiated and written to output_filename.

    Returns
    -------
    matplotlib.axes.Axes or None
        Returns the active axes object if an ax argument was injected; otherwise
        saves the figure to disk and returns None.
    """
    if style_file and os.path.exists(style_file):
        plt.style.use(style_file)

    figsize = (6, 6)
    vbm_cbm_color = {"vbm": "orange", "cbm": "green", "alpha": 0.5}

    standalone = ax is None
    if standalone:
        fig, target_ax = plt.subplots(figsize=figsize)
    else:
        target_ax = ax

    target_ax.set_xlim(-data.lim, data.lim)
    target_ax.set_ylim(data.emin, data.emax)
    target_ax.axvline(0, color="black", linestyle="--", alpha=0.5)

    # Shade bulk band edges regions
    target_ax.axhspan(
        data.emin, data.vbm, color=vbm_cbm_color["vbm"], alpha=vbm_cbm_color["alpha"]
    )
    target_ax.axhspan(
        data.cbm, data.emax, color=vbm_cbm_color["cbm"], alpha=vbm_cbm_color["alpha"]
    )

    # Adjust marker line layout scaling dynamically
    s = data.w * 200 * (figsize[0] / 6.0)
    target_ax.scatter(data.xvalues_up, data.up_energies, color="black", marker="_", s=s)
    target_ax.scatter(
        data.xvalues_down, data.down_energies, color="black", marker="_", s=s
    )

    electron_markers = {"occupied": "o", "unoccupied": "x", "s": s / 25.0}

    # Draw electron occupation dots and holes representations
    target_ax.scatter(
        data.occupied_up["xvalues"],
        data.occupied_up["energies"],
        color="k",
        marker=electron_markers["occupied"],
        s=electron_markers["s"],
    )
    target_ax.scatter(
        data.unoccupied_up["xvalues"],
        data.unoccupied_up["energies"],
        color="k",
        marker=electron_markers["unoccupied"],
        s=electron_markers["s"],
    )
    target_ax.scatter(
        data.occupied_down["xvalues"],
        data.occupied_down["energies"],
        color="k",
        marker=electron_markers["occupied"],
        s=electron_markers["s"],
    )
    target_ax.scatter(
        data.unoccupied_down["xvalues"],
        data.unoccupied_down["energies"],
        color="k",
        marker=electron_markers["unoccupied"],
        s=electron_markers["s"],
    )

    target_ax.set_ylabel("Energy (eV)")
    target_ax.set_xticks([])
    target_ax.set_xticklabels([])

    if standalone:
        plt.savefig(Path(output_filename), dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return target_ax


def _lookup_pr_values(
    pr_result: dict,
    band_indices: List[int],
    spin_label: str,
    metric: str,
    kpt_idx: int = 0,
) -> List[float]:
    """
    Look up metric values from *pr_result* for a list of 0-based band indices.

    Returns NaN for any band that is not found in the result.
    """
    kpt_label  = f"kpt_{kpt_idx + 1}"
    spin_block = pr_result.get("data", {}).get(spin_label, {})
    band_block = spin_block.get(kpt_label, {})
    out = []
    for bi in band_indices:
        key = f"band_{bi + 1}"
        val = band_block.get(key, {}).get(metric)
        out.append(float(val) if val is not None else float("nan"))
    return out


def plot_ks_with_pr(
    ks_data: KohnShamPlotData,
    pr_result: dict,
    metric: str = "p_ratio",
    cmap: str = "RdYlGn_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    threshold: Optional[float] = None,
    kpt_idx: int = 0,
    title: Optional[str] = None,
    output_filename: Union[str, Path] = "ks_pr_plot.png",
    figsize: Tuple[float, float] = (7, 6),
    dpi: int = 300,
    style_file: Optional[str] = None,
) -> None:
    """
    Plot Kohn-Sham energy levels colour-coded by participation ratio (P-ratio or IPR).

    The layout follows :func:`plot_spin_resolved_levels` (spin-up on the left,
    spin-down on the right with a dividing vertical dashed line) but each
    horizontal level bar is coloured by the *metric* value fetched from
    *pr_result* instead of being drawn in uniform black.

    Parameters
    ----------
    ks_data : KohnShamPlotData
        Processed eigenvalue data container (from :func:`extract_ksplot_data`).
    pr_result : dict
        Nested participation-ratio result dict (loaded from
        ``participation_ratio.json`` or returned by
        :func:`compute_participation_ratios`).
    metric : {"p_ratio", "ipr"}
        The localization metric used for coloring.  Default ``"p_ratio"``.
    cmap : str
        Matplotlib colormap name.  Default ``"RdYlGn_r"`` (green = low, red = high).
    vmin : float
        Colormap lower bound.  Default 0.0.
    vmax : float
        Colormap upper bound.  Default 1.0.
    threshold : float, optional
        If provided, draw a horizontal dashed line at this metric value to
        guide the eye — only meaningful when a dual-axis view is used.
        (Currently the colour already encodes the metric, so this is optional.)
    kpt_idx : int
        0-based k-point index to use when looking up PR values.  Default 0.
    title : str, optional
        Figure title.  Defaults to the defect name stored in *pr_result*.
    output_filename : str or Path
        Destination file path.  Default ``"ks_pr_plot.png"``.
    figsize : tuple of float
        Figure size (width, height) in inches.
    dpi : int
        Image resolution.
    style_file : str, optional
        Optional matplotlib ``.mplstyle`` file path.

    Returns
    -------
    None
        Saves the figure to *output_filename*.
    """
    if style_file and os.path.exists(style_file):
        plt.style.use(style_file)

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
    import matplotlib as _mpl
    try:
        colormap = _mpl.colormaps[cmap]
    except AttributeError:
        colormap = cm.get_cmap(cmap)  # matplotlib < 3.5 fallback
    scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_map.set_array([])

    defect_name = pr_result.get("defect_name", "defect")

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, target_ax = plt.subplots(figsize=figsize)

    target_ax.set_xlim(-ks_data.lim, ks_data.lim)
    target_ax.set_ylim(ks_data.emin, ks_data.emax)
    target_ax.axvline(0, color="black", linestyle="--", alpha=0.5)

    vbm_cbm_color = {"vbm": "orange", "cbm": "green", "alpha": 0.35}
    target_ax.axhspan(ks_data.emin, ks_data.vbm,
                      color=vbm_cbm_color["vbm"], alpha=vbm_cbm_color["alpha"])
    target_ax.axhspan(ks_data.cbm, ks_data.emax,
                      color=vbm_cbm_color["cbm"], alpha=vbm_cbm_color["alpha"])

    s = ks_data.w * 200 * (figsize[0] / 6.0)

    # ── spin-up (left, spin_1) ────────────────────────────────────────────────
    up_pr = _lookup_pr_values(
        pr_result, ks_data.up_idx, "spin_1", metric, kpt_idx=kpt_idx
    )
    up_colors = [
        scalar_map.to_rgba(v) if not (v != v) else "lightgray"   # NaN → gray
        for v in up_pr
    ]
    target_ax.scatter(
        ks_data.xvalues_up, ks_data.up_energies,
        c=up_colors, marker="_", s=s, zorder=3, linewidths=2,
    )

    # ── spin-down (right, spin_2) ─────────────────────────────────────────────
    spin2_label = "spin_2" if "spin_2" in pr_result.get("data", {}) else "spin_1"
    down_pr = _lookup_pr_values(
        pr_result, ks_data.down_idx, spin2_label, metric, kpt_idx=kpt_idx
    )
    down_colors = [
        scalar_map.to_rgba(v) if not (v != v) else "lightgray"
        for v in down_pr
    ]
    target_ax.scatter(
        ks_data.xvalues_down, ks_data.down_energies,
        c=down_colors, marker="_", s=s, zorder=3, linewidths=2,
    )

    # ── occupation markers ────────────────────────────────────────────────────
    em = s / 25.0
    for xv, en, occ in zip(ks_data.xvalues_up, ks_data.up_energies,
                            ks_data.up_occupations):
        mk = "o" if occ > 0.6 else "x"
        target_ax.scatter([xv], [en], color="k", marker=mk, s=em, zorder=4)
    for xv, en, occ in zip(ks_data.xvalues_down, ks_data.down_energies,
                            ks_data.down_occupations):
        mk = "o" if occ > 0.6 else "x"
        target_ax.scatter([xv], [en], color="k", marker=mk, s=em, zorder=4)

    # ── labels & colorbar ─────────────────────────────────────────────────────
    metric_label = {"p_ratio": "P-ratio", "ipr": "IPR"}.get(metric, metric)
    cbar = fig.colorbar(scalar_map, ax=target_ax, pad=0.02, fraction=0.04)
    cbar.set_label(metric_label, fontsize=9)

    target_ax.set_ylabel("Energy (eV)")
    target_ax.set_xticks([])
    target_ax.set_xticklabels([])

    # Spin-channel labels below x-axis
    target_ax.text(-ks_data.lim / 2, ks_data.emin, "spin ↑",
                   ha="center", va="bottom", fontsize=8, color="gray")
    target_ax.text( ks_data.lim / 2, ks_data.emin, "spin ↓",
                   ha="center", va="bottom", fontsize=8, color="gray")

    target_ax.set_title(title or defect_name)

    fig.tight_layout()
    fig.savefig(Path(output_filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
