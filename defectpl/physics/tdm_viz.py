# -*- coding: utf-8 -*-
"""
defectpl.physics.tdm_viz
========================
Visualisation utilities for TDM / IPR results and wavefunction export.

TDM plots
---------
* :func:`plot_tdm_heatmap`      — |TDM| between band pairs at all k-points
* :func:`plot_tdm_bubble`       — bubble chart of |TDM| vs ΔE
* :func:`plot_tdm_components`   — x/y/z TDM components vs k-index
* :func:`plot_tdm_kpoint_strip` — |TDM| along a k-path strip
* :func:`plot_tdm_absorption`   — Gaussian-broadened absorption spectrum
* :func:`plot_tdm_dashboard`    — 2×3 summary dashboard

IPR plots
---------
* :func:`plot_ipr_scatter`         — IPR vs eigenvalue
* :func:`plot_ipr_bar`             — bar chart of band-averaged IPR
* :func:`plot_ipr_kpoint_heatmap`  — IPR across (band × k-point)

Wavefunction export
-------------------
* :func:`save_wfc_vasp`   — CHGCAR-format real-space wavefunction (VESTA / VESTA)
* :func:`save_wfc_vesta`  — .vesta project file for one-click VESTA visualisation

All matplotlib functions accept a ``fig_kwargs`` dict that is passed to
``plt.subplots()``.  To override resolution, size, or backend, pass keyword
arguments there (e.g. ``fig_kwargs={"figsize": (10, 6), "dpi": 200}``).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # TDM plots
    "plot_tdm_heatmap",
    "plot_tdm_bubble",
    "plot_tdm_components",
    "plot_tdm_kpoint_strip",
    "plot_tdm_absorption",
    "plot_tdm_dashboard",
    # IPR plots
    "plot_ipr_scatter",
    "plot_ipr_bar",
    "plot_ipr_kpoint_heatmap",
    # WFC export
    "save_wfc_vasp",
    "save_wfc_vesta",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_mpl():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for TDM plots. Install with: pip install matplotlib"
        )


def _save_or_show(fig, outfile: Optional[Union[str, Path]]) -> None:
    """Save to file or display figure."""
    plt = _import_mpl()
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Figure saved to: {outfile}")
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================
# TDM visualisation
# ===========================================================================


def plot_tdm_heatmap(
    result: dict,
    component: str = "magnitude",
    cmap: str = "viridis",
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """Heatmap of |TDM| vs k-point for a single band pair.

    Parameters
    ----------
    result : dict
        Output of :meth:`~defectpl.physics.tdm.WavecarReader.get_tdm_all_kpoints`.
    component : {'magnitude', 'x', 'y', 'z'}
        Which TDM quantity to display.
    cmap : str
    outfile : str or Path, optional
        Save path.  If None, the figure is displayed interactively.
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> res = wfc.get_tdm_all_kpoints(1, 638, 639)
    >>> plot_tdm_heatmap(res, outfile="heatmap.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (8, 4))
    fig, ax = plt.subplots(**fk)

    if component == "magnitude":
        values = result["tdm_magnitude"]
        title = f"TDM |μ| (Debye)  —  bands {result['iband_i']}→{result['iband_j']}"
    elif component in ("x", "y", "z"):
        idx = {"x": 0, "y": 1, "z": 2}[component]
        values = result["tdm_components"][:, idx]
        title = f"TDM μ_{component} (Debye)  —  bands {result['iband_i']}→{result['iband_j']}"
    else:
        raise ValueError(
            f"component must be 'magnitude', 'x', 'y', or 'z'. Got '{component}'."
        )

    nkpts = len(values)
    im = ax.imshow(
        values[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        extent=[0.5, nkpts + 0.5, 0, 1],
    )
    fig.colorbar(im, ax=ax, label="|μ| (Debye)")
    ax.set_xlabel("k-point index")
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_tdm_bubble(
    result: dict,
    scale: float = 3000.0,
    color: str = "tab:blue",
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """Bubble chart of |TDM| vs ΔE across k-points.

    Parameters
    ----------
    result : dict
        Output of :meth:`~defectpl.physics.tdm.WavecarReader.get_tdm_all_kpoints`.
    scale : float
        Bubble area scale factor.
    color : str
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_tdm_bubble(res, outfile="bubble.pdf")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (7, 5))
    fig, ax = plt.subplots(**fk)

    dE = result["dE"]
    mag = result["tdm_magnitude"]
    ax.scatter(
        dE, mag, s=mag**2 * scale, alpha=0.6, c=color, edgecolors="k", linewidths=0.5
    )
    ax.set_xlabel("ΔE (eV)")
    ax.set_ylabel("|TDM| (Debye)")
    ax.set_title(f"TDM bubble chart  —  bands {result['iband_i']}→{result['iband_j']}")
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_tdm_components(
    result: dict,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """Line plot of |μ_x|, |μ_y|, |μ_z|, and |μ| vs k-index.

    Parameters
    ----------
    result : dict
        Output of :meth:`~defectpl.physics.tdm.WavecarReader.get_tdm_all_kpoints`.
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_tdm_components(res, outfile="components.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (9, 5))
    fig, ax = plt.subplots(**fk)

    kpts = np.arange(1, len(result["dE"]) + 1)
    comp = result["tdm_components"]
    mag = result["tdm_magnitude"]

    for ci, (label, ls) in enumerate(
        zip(
            ["|μ_x|", "|μ_y|", "|μ_z|"],
            ["--", "-.", ":"],
        )
    ):
        ax.plot(kpts, comp[:, ci], ls=ls, label=label, lw=1.2)
    ax.plot(kpts, mag, "k-", label="|μ| total", lw=2)

    ax.set_xlabel("k-point index")
    ax.set_ylabel("|TDM| (Debye)")
    ax.set_title(f"TDM components  —  bands {result['iband_i']}→{result['iband_j']}")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_tdm_kpoint_strip(
    result: dict,
    kpath_labels: Optional[Dict[int, str]] = None,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """|TDM| as a strip chart along a k-point path.

    Parameters
    ----------
    result : dict
    kpath_labels : dict, optional
        Mapping ``{kpt_index_1based: "Γ"}`` for high-symmetry labels.
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (10, 4))
    fig, ax = plt.subplots(**fk)

    kpts = np.arange(1, len(result["tdm_magnitude"]) + 1)
    ax.fill_between(kpts, result["tdm_magnitude"], alpha=0.4)
    ax.plot(kpts, result["tdm_magnitude"], lw=1.5)
    ax.set_xlabel("k-point index")
    ax.set_ylabel("|TDM| (Debye)")
    ax.set_title(f"TDM strip  —  bands {result['iband_i']}→{result['iband_j']}")

    if kpath_labels:
        for ki, label in kpath_labels.items():
            ax.axvline(ki, color="gray", lw=0.8, ls="--")
            ax.text(
                ki, ax.get_ylim()[1] * 0.95, label, ha="center", va="top", fontsize=9
            )

    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_tdm_absorption(
    result: dict,
    sigma: float = 0.05,
    n_pts: int = 500,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Gaussian-broadened mock absorption spectrum weighted by |TDM|².

    Parameters
    ----------
    result : dict
        Output of :meth:`~defectpl.physics.tdm.WavecarReader.get_tdm_all_kpoints`.
    sigma : float — Gaussian broadening in eV.
    n_pts : int — number of energy points.
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    (fig, energies, spectrum) : tuple
        Figure, x array (eV), and broadened spectrum (arb. units).

    Examples
    --------
    >>> fig, e, spec = plot_tdm_absorption(res, sigma=0.05, outfile="abs.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (8, 5))
    fig, ax = plt.subplots(**fk)

    dE = result["dE"]
    mag = result["tdm_magnitude"]
    E_lo, E_hi = dE.min() - 3 * sigma, dE.max() + 3 * sigma
    E_grid = np.linspace(E_lo, E_hi, n_pts)
    spectrum = np.zeros(n_pts)
    for e_k, mu_k in zip(dE, mag):
        spectrum += mu_k**2 * np.exp(-0.5 * ((E_grid - e_k) / sigma) ** 2)
    spectrum /= spectrum.max() if spectrum.max() > 0 else 1.0

    ax.plot(E_grid, spectrum, lw=2)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title(
        f"Mock absorption  —  bands {result['iband_i']}→{result['iband_j']}, σ={sigma} eV"
    )
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig, E_grid, spectrum


def plot_tdm_dashboard(
    result: dict,
    sigma: float = 0.05,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """2×3 summary dashboard: components, bubble, heatmap, absorption, strip, stats.

    Parameters
    ----------
    result : dict
        Output of :meth:`~defectpl.physics.tdm.WavecarReader.get_tdm_all_kpoints`.
    sigma : float — Gaussian broadening for absorption panel.
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_tdm_dashboard(res, outfile="dashboard.pdf")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (15, 8))
    fig, axes = plt.subplots(2, 3, **fk)
    fig.suptitle(
        f"TDM Dashboard  —  spin {result['ispin']}  "
        f"bands {result['iband_i']}→{result['iband_j']}",
        fontsize=13,
    )

    kpts = np.arange(1, len(result["dE"]) + 1)
    comp = result["tdm_components"]
    mag = result["tdm_magnitude"]
    dE = result["dE"]

    ax = axes[0, 0]
    for ci, (label, ls) in enumerate(
        zip(["|μ_x|", "|μ_y|", "|μ_z|"], ["--", "-.", ":"])
    ):
        ax.plot(kpts, comp[:, ci], ls=ls, label=label, lw=1.2)
    ax.plot(kpts, mag, "k-", label="|μ|", lw=2)
    ax.set_xlabel("k-index")
    ax.set_ylabel("|TDM| (D)")
    ax.legend(fontsize=7)
    ax.set_title("Components")

    ax = axes[0, 1]
    ax.scatter(
        dE, mag, s=mag**2 * 2000, alpha=0.6, c="tab:blue", edgecolors="k", lw=0.5
    )
    ax.set_xlabel("ΔE (eV)")
    ax.set_ylabel("|TDM| (D)")
    ax.set_title("Bubble chart")

    ax = axes[0, 2]
    ax.imshow(
        mag[np.newaxis, :],
        aspect="auto",
        cmap="viridis",
        extent=[0.5, len(mag) + 0.5, 0, 1],
    )
    ax.set_xlabel("k-index")
    ax.set_yticks([])
    ax.set_title("Heatmap")

    ax = axes[1, 0]
    E_lo, E_hi = dE.min() - 3 * sigma, dE.max() + 3 * sigma
    E_grid = np.linspace(E_lo, E_hi, 500)
    spectrum = sum(
        mu**2 * np.exp(-0.5 * ((E_grid - ek) / sigma) ** 2) for ek, mu in zip(dE, mag)
    )
    mx = spectrum.max()
    if mx > 0:
        spectrum /= mx
    ax.plot(E_grid, spectrum, lw=2)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Abs. (arb.)")
    ax.set_title(f"Absorption σ={sigma} eV")

    ax = axes[1, 1]
    ax.fill_between(kpts, mag, alpha=0.4)
    ax.plot(kpts, mag, lw=1.5)
    ax.set_xlabel("k-index")
    ax.set_ylabel("|TDM| (D)")
    ax.set_title("Strip")

    ax = axes[1, 2]
    stats = {
        "Mean |μ|": f"{mag.mean():.3f} D",
        "Max |μ|": f"{mag.max():.3f} D",
        "Min |μ|": f"{mag.min():.3f} D",
        "Std |μ|": f"{mag.std():.3f} D",
        "Mean ΔE": f"{dE.mean():.3f} eV",
    }
    ax.axis("off")
    text = "\n".join(f"{k:>12s}  {v}" for k, v in stats.items())
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
    )
    ax.set_title("Statistics")

    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


# ===========================================================================
# IPR visualisation
# ===========================================================================


def plot_ipr_scatter(
    result: dict,
    fermi_level: float = 0.0,
    use_weighted: bool = True,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """Scatter plot: IPR vs eigenvalue for all selected bands.

    Parameters
    ----------
    result : dict
        Output of :func:`~defectpl.physics.tdm.compute_ipr_all`.
    fermi_level : float — shift eigenvalues so E_F = 0.
    use_weighted : bool — use k-weighted average IPR.
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> ipr_result = compute_ipr_all(wfc, 1, kweights=kw)
    >>> plot_ipr_scatter(ipr_result, fermi_level=ef, outfile="ipr_scatter.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    fk.setdefault("figsize", (8, 5))
    fig, ax = plt.subplots(**fk)

    col_key = "weighted_avg_ipr" if use_weighted else "avg_ipr"
    energies = np.array([r["avg_energy"] for r in result["band_summary"]]) - fermi_level
    iprs = np.array([r[col_key] for r in result["band_summary"]])

    scatter = ax.scatter(energies, iprs, c=iprs, cmap="hot_r", edgecolors="k", lw=0.4)
    fig.colorbar(scatter, ax=ax, label="IPR")
    ax.axvline(0.0, color="gray", lw=1.0, ls="--", label="$E_F$")
    ax.set_xlabel("$E - E_F$ (eV)")
    ax.set_ylabel("IPR")
    label = "k-weighted avg IPR" if use_weighted else "avg IPR"
    ax.set_title(f"IPR vs eigenvalue  ({label})")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_ipr_bar(
    result: dict,
    top_n: int = 20,
    use_weighted: bool = True,
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """Horizontal bar chart of the bands with the highest IPR.

    Parameters
    ----------
    result : dict — from :func:`~defectpl.physics.tdm.compute_ipr_all`.
    top_n : int — number of bands to show.
    use_weighted : bool
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_ipr_bar(ipr_result, top_n=15, outfile="ipr_bar.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}
    col_key = "weighted_avg_ipr" if use_weighted else "avg_ipr"

    summary_sorted = sorted(
        result["band_summary"], key=lambda r: r[col_key], reverse=True
    )[:top_n]

    labels = [f"Band {r['iband']}" for r in summary_sorted]
    iprs = [r[col_key] for r in summary_sorted]

    fk.setdefault("figsize", (6, max(3, 0.35 * top_n)))
    fig, ax = plt.subplots(**fk)

    ax.barh(labels[::-1], iprs[::-1], color="tab:blue", edgecolor="k", lw=0.5)
    ax.set_xlabel("IPR")
    ax.set_title(f"Top-{top_n} bands by IPR")
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


def plot_ipr_kpoint_heatmap(
    result: dict,
    bands: Optional[Sequence[int]] = None,
    cmap: str = "magma",
    outfile: Optional[Union[str, Path]] = None,
    fig_kwargs: Optional[dict] = None,
) -> Any:
    """2-D heatmap of IPR across (band × k-point).

    Parameters
    ----------
    result : dict — from :func:`~defectpl.physics.tdm.compute_ipr_all`.
    bands : sequence of int, optional — subset of band indices to show.
    cmap : str
    outfile : str or Path, optional
    fig_kwargs : dict, optional

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_ipr_kpoint_heatmap(ipr_result, outfile="ipr_heatmap.png")
    """
    plt = _import_mpl()
    fk = fig_kwargs or {}

    meta = result["metadata"]
    nkpts = meta["nkpts"]
    all_b = meta["bands"]
    if bands is not None:
        bands_show = [b for b in all_b if b in set(bands)]
    else:
        bands_show = all_b

    ipr_map = {(r["iband"], r["ikpt"]): r["ipr"] for r in result["per_band_per_kpoint"]}
    matrix = np.array(
        [[ipr_map.get((b, k), 0.0) for k in range(1, nkpts + 1)] for b in bands_show]
    )

    fk.setdefault("figsize", (max(6, nkpts * 0.3), max(4, len(bands_show) * 0.4)))
    fig, ax = plt.subplots(**fk)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
    fig.colorbar(im, ax=ax, label="IPR")
    ax.set_yticks(range(len(bands_show)))
    ax.set_yticklabels([str(b) for b in bands_show], fontsize=7)
    ax.set_xlabel("k-point index")
    ax.set_ylabel("Band index")
    ax.set_title("IPR heatmap (band × k-point)")
    fig.tight_layout()
    _save_or_show(fig, outfile)
    return fig


# ===========================================================================
# Real-space wavefunction export
# ===========================================================================


def _write_chgcar_header(
    fout: io.TextIOWrapper,
    lattice: np.ndarray,
    species: List[str],
    counts: List[int],
    positions: np.ndarray,
    ngrid: Tuple[int, int, int],
    title: str = "Wavefunction",
) -> None:
    """Write POSCAR-like header section of a CHGCAR file."""
    fout.write(f"{title}\n")
    fout.write("  1.0\n")
    for row in lattice:
        fout.write(f"  {row[0]: .10f}  {row[1]: .10f}  {row[2]: .10f}\n")
    fout.write("  " + "  ".join(species) + "\n")
    fout.write("  " + "  ".join(str(c) for c in counts) + "\n")
    fout.write("Direct\n")
    for pos in positions:
        fout.write(f"  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")
    fout.write("\n")
    fout.write(f"  {ngrid[0]}  {ngrid[1]}  {ngrid[2]}\n")


def save_wfc_vasp(
    wfc,
    ispin: int,
    ikpt: int,
    iband: int,
    structure: dict,
    ngrid=None,
    outfile: Union[str, Path] = "WAVECAR_density.vasp",
    quantity: str = "density",
) -> None:
    """Export real-space KS state to CHGCAR format for VESTA.

    Parameters
    ----------
    wfc : WavecarReader or compatible
    ispin, ikpt, iband : int — 1-based.
    structure : dict
        From :func:`~defectpl.io.wavecar.read_poscar` or
        :func:`~defectpl.io.wavecar.get_structure`.
    ngrid : array-like of int, optional — FFT grid.
    outfile : str or Path
    quantity : {'density', 'real', 'imag'}
        ``'density'``: |φ|², ``'real'``: Re(φ), ``'imag'``: Im(φ).

    Examples
    --------
    >>> struct = get_structure("/data/ground/")
    >>> save_wfc_vasp(wfc, 1, 1, 638, struct, outfile="band638.vasp")
    """
    phi = wfc.wfc_r(ispin, ikpt, iband, ngrid=ngrid)
    if isinstance(phi, list):
        phi_combined = sum(np.abs(p) ** 2 for p in phi)
        data_arr = phi_combined
        title = f"WFC band {iband} (SOC spinor density)"
    else:
        if quantity == "density":
            data_arr = np.abs(phi) ** 2
            title = f"|psi|^2 band {iband} k{ikpt} spin{ispin}"
        elif quantity == "real":
            data_arr = phi.real
            title = f"Re(psi) band {iband} k{ikpt} spin{ispin}"
        elif quantity == "imag":
            data_arr = phi.imag
            title = f"Im(psi) band {iband} k{ikpt} spin{ispin}"
        else:
            raise ValueError(
                f"quantity must be 'density', 'real', or 'imag'. Got '{quantity}'."
            )

    nx, ny, nz = data_arr.shape
    ngrid_used = (nx, ny, nz)
    volume = abs(float(np.linalg.det(structure["lattice"])))
    data_arr = data_arr / volume

    with open(outfile, "w", encoding="utf-8") as fout:
        _write_chgcar_header(
            fout,
            lattice=structure["lattice"],
            species=structure["species"],
            counts=structure["counts"],
            positions=structure["positions"],
            ngrid=ngrid_used,
            title=title,
        )
        flat = data_arr.flatten(order="F")
        ncols = 5
        for i in range(0, len(flat), ncols):
            row = flat[i : i + ncols]
            fout.write("  " + "  ".join(f"{v:.10E}" for v in row) + "\n")

    mb = Path(outfile).stat().st_size / 1e6
    print(f"CHGCAR wavefunction written to: {outfile}  ({mb:.2f} MB)")


def save_wfc_vesta(
    wfc,
    ispin: int,
    ikpt: int,
    iband: int,
    structure: dict,
    vasp_file: Optional[Union[str, Path]] = None,
    ngrid=None,
    outfile: Union[str, Path] = "WAVECAR_density.vesta",
    isosurface: float = 0.001,
) -> None:
    """Write a VESTA project file (.vesta) for one-click wavefunction visualisation.

    Optionally also writes the CHGCAR file if ``vasp_file`` is not provided.

    Parameters
    ----------
    wfc : WavecarReader or compatible
    ispin, ikpt, iband : int — 1-based.
    structure : dict
    vasp_file : str or Path, optional
        Pre-existing CHGCAR file.  Written automatically if not given.
    ngrid : optional
    outfile : str or Path
    isosurface : float — isosurface value in the VESTA file.

    Examples
    --------
    >>> save_wfc_vesta(wfc, 1, 1, 638, struct, outfile="band638.vesta")
    """
    if vasp_file is None:
        vasp_file = Path(outfile).with_suffix(".vasp")
        save_wfc_vasp(
            wfc, ispin, ikpt, iband, structure, ngrid=ngrid, outfile=vasp_file
        )
    vasp_file = Path(vasp_file)
    outfile = Path(outfile)

    lat = structure["lattice"]
    species_counts = list(zip(structure["species"], structure["counts"]))
    pos_frac = structure["positions"]

    from io import StringIO

    buf = StringIO()
    buf.write("#VESTA_FORMAT_VERSION 3.5.0\n\n")
    buf.write("CRYSTAL\n\nTITLE\n")
    buf.write(f"Band {iband} k={ikpt} spin={ispin}\n\n")
    buf.write("CELLP\n")
    a_len = float(np.linalg.norm(lat[0]))
    b_len = float(np.linalg.norm(lat[1]))
    c_len = float(np.linalg.norm(lat[2]))
    al = float(np.degrees(np.arccos(np.dot(lat[1], lat[2]) / (b_len * c_len))))
    be = float(np.degrees(np.arccos(np.dot(lat[0], lat[2]) / (a_len * c_len))))
    ga = float(np.degrees(np.arccos(np.dot(lat[0], lat[1]) / (a_len * b_len))))
    buf.write(
        f"  {a_len:.6f}  {b_len:.6f}  {c_len:.6f}  {al:.4f}  {be:.4f}  {ga:.4f}\n"
    )
    buf.write("  0 0 0 0 0 0\n\n")
    buf.write("STRUC\n")
    atom_num = 1
    for sp, count in species_counts:
        for i in range(count):
            p = pos_frac[atom_num - 1]
            buf.write(
                f"  {atom_num}  {sp}  {sp}{atom_num}  1.0000  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}   1  -\n"
            )
            atom_num += 1
    buf.write("  0 0 0 0 0 0 0\n\n")
    buf.write(
        "VECTR\n 0 0 0 0 0\n\nVECTT\n 0 0 0 0 0\n\nSPLAN\n  0   0   0   0\n\nLBLAT\n -1\n\n"
    )
    buf.write("LBTYP\n  0\n\nATMOV\n  0 0 0 0\n\n")
    buf.write("BONDS\n 0 0 0 0 0 0 0 0 0 0 0 0\n\n")
    buf.write("SBOND\n 0 0 0 0 0 0\n\n")
    buf.write("SITET\n 0 0 0 0 0 0 0\n\n")
    buf.write("VECTT\n 0 0 0 0 0\n\n")
    buf.write(f"ISOSURF\n  {isosurface:.5f}  2   2  0  0  0\n\n")
    buf.write("IMPORT_DENSITY 1\n")
    buf.write(f"+1.000000  {vasp_file.name}\n\n")
    buf.write("END\n")

    with open(outfile, "w") as fout:
        fout.write(buf.getvalue())
    print(f"VESTA project file written to: {outfile}")
