# -*- coding: utf-8 -*-
"""
Stores publication-grade plotting functions for the defectpl package.
Author : Shibu Meher
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monty.serialization import loadfn

from defectpl.constants import EV2MEV

# Load custom publication style with a robust procedural fallback layout
style_file = Path(__file__).parent / "defectpl.mplstyle"
if style_file.exists():
    style.use(str(style_file))
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


class Plotter:
    """Handles static publication-quality matplotlib figures."""

    def __init__(self):
        pass

    def _save_or_show(self, fig, out_dir: Union[str, Path], file_name: str, fig_format: str, plot: bool):
        """Internal helper to streamline image exportation or execution visualization."""
        if plot:
            plt.show()
            plt.close(fig)
        else:
            valid_formats = ["png", "pdf", "svg", "jpg", "jpeg"]
            fmt = fig_format.lower() if fig_format.lower() in valid_formats else "pdf"
            
            # FIXED: Changed {fig_format} to {fmt}
            out_path = Path(out_dir) / f"{file_name}.{fmt}" 
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(out_path, dpi=600, bbox_inches="tight", format=fmt)
            plt.close(fig)

    def plot_penergy_vs_pmode(
        self,
        frequencies: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "penergy_vs_pmode",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots phonon energy vs phonon mode index."""
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0
        mode_idx = np.arange(1, len(freq_mev) + 1)
        
        ax.plot(mode_idx, freq_mev, color="black")
        ax.set_xlabel("Phonon Mode Index")
        ax.set_ylabel("Phonon Energy (meV)")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_ipr_vs_penergy(
        self,
        frequencies: np.ndarray,
        iprs: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "ipr_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots Inverse Participation Ratio (IPR) vs phonon energy."""
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0
        
        ax.scatter(freq_mev, iprs, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel("IPR")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_loc_rat_vs_penergy(
        self,
        frequencies: np.ndarray,
        localization_ratio: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "loc_rat_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots Mode Localization Ratio vs phonon energy."""
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0
        
        ax.scatter(freq_mev, localization_ratio, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel("Localization Ratio")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_qk_vs_penergy(
        self,
        frequencies: np.ndarray,
        qks: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "qk_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots configurational vibrational mode displacement (q_k) vs phonon energy."""
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0
        
        ax.scatter(freq_mev, qks, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(r"Displacement $q_k$ ($\mathrm{amu}^{1/2}\cdot\mathrm{\AA}$)")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_HR_factor_vs_penergy(
        self,
        frequencies: np.ndarray,
        Sks: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "HR_factor_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots partial mode Huang-Rhys factors (S_k) vs phonon energy."""
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0
        
        ax.scatter(freq_mev, Sks, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(r"Partial HR Factor ($S_k$)")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_S_omega_vs_penergy(
        self,
        frequencies: np.ndarray,
        S_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "S_omega_vs_penergy",
        max_freq: Optional[float] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots continuous Huang-Rhys spectral density S(omega) distribution."""
        fig, ax = plt.subplots(figsize=figsize)
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
        
        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff
        
        S_omega_mev = np.array(S_omega)[mask] / 1000.0
        energies_mev = omega_set[mask] * 1000.0
        
        ax.plot(energies_mev, S_omega_mev, color="black")
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)")
        ax.set_xlim(0, cutoff * 1000.0)
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_S_omega_Sks_vs_penergy(
        self,
        frequencies: np.ndarray,
        S_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        Sks: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "S_omega_Sks_vs_penergy",
        max_freq: Optional[float] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.5, 2.5),
    ):
        """Plots continuous S(omega) and partial mode S_k factors overlaid using dual axes."""
        fig, ax1 = plt.subplots(figsize=figsize)
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
        
        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff
        
        # Pull standard colors directly from cycler layout
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1, c2 = colors[2], colors[0]  # Safe default color indices mapping
        
        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color=c1)
        ax1.plot(omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color=c1)
        ax1.tick_params(axis="y", labelcolor=c1)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color=c2)
        ax2.scatter(np.array(frequencies) * 1000.0, Sks, color=c2, edgecolor="black", alpha=0.85)
        ax2.tick_params(axis="y", labelcolor=c2)
        
        ax1.set_xlim(0, cutoff * 1000.0)
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_S_omega_Sks_Loc_rat_vs_penergy(
        self,
        frequencies: np.ndarray,
        S_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        Sks: np.ndarray,
        localization_ratio: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "S_omega_HRf_loc_rat_vs_penergy",
        max_freq: Optional[float] = None,
        pylim: List[Optional[float]] = [None, None],
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (4.2, 2.5),
        cmap: str = "viridis",
    ):
        """Plots S(omega) curve and scattered partial S_k points color-mapped by Localization Ratio."""
        # FIX: Explicitly set layout="constrained" to cleanly handle twinx + colorbar spacing
        fig, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
        
        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff
        
        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color="black")
        ax1.plot(omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color="black")
        
        sc = ax2.scatter(
            np.array(frequencies) * 1000.0, 
            Sks, 
            c=localization_ratio, 
            cmap=plt.get_cmap(cmap), 
            edgecolor="black", 
            alpha=0.85
        )
        ax2.tick_params(axis="y", labelcolor="black")
        if pylim[0] is not None or pylim[1] is not None:
            ax2.set_ylim(pylim[0], pylim[1])
            
        ax1.set_xlim(0, cutoff * 1000.0)
        
        # When using layout="constrained", we can target ax2 directly with an explicit pad
        cbar = fig.colorbar(sc, ax=ax2, pad=0.12)
        cbar.set_label("Localization Ratio")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_S_omega_Sks_ipr_vs_penergy(
        self,
        frequencies: np.ndarray,
        S_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        Sks: np.ndarray,
        iprs: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "S_omega_HRf_ipr_vs_penergy",
        max_freq: Optional[float] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (4.2, 2.5),
        cmap: str = "viridis",
    ):
        """Plots S(omega) curve and scattered partial S_k points color-mapped by IPR values."""
        # FIX: Explicitly set layout="constrained" to cleanly handle twinx + colorbar spacing
        fig, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
        
        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff
        
        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color="black")
        ax1.plot(omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color="black")
        
        sc = ax2.scatter(
            np.array(frequencies) * 1000.0, 
            Sks, 
            c=iprs, 
            cmap=plt.get_cmap(cmap), 
            edgecolor="black", 
            alpha=0.85
        )
        ax2.tick_params(axis="y", labelcolor="black")
        ax1.set_xlim(0, cutoff * 1000.0)
        
        # When using layout="constrained", we can target ax2 directly with an explicit pad
        cbar = fig.colorbar(sc, ax=ax2, pad=0.12)
        cbar.set_label("Inverse Participation Ratio")
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_intensity_vs_penergy(
        self,
        frequencies: np.ndarray,
        I: np.ndarray,
        resolution: int,
        xlim: Tuple[float, float],
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "intensity_vs_penergy",
        iylim: Optional[Tuple[float, float]] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """Plots normalized photoluminescence intensity against emission energy."""
        fig, ax = plt.subplots(figsize=figsize)
        
        x_energy_ev = np.arange(len(I)) / float(resolution)
        I_abs = np.abs(I)
        I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs
        
        ax.plot(x_energy_ev, I_norm, color="black")
        ax.set_ylabel("PL Intensity (arb. u.)")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_xlim(xlim[0], xlim[1])
        
        if iylim:
            ax.set_ylim(iylim[0], iylim[1])
        ax.set_yticks([])  # Clean look for publication spectra
        
        self._save_or_show(fig, out_dir, file_name, fig_format, plot)


# =====================================================================
# Standalone Interactive Plotly & Comparison Utilities
# =====================================================================

def plot_interactive_intensity(filename: Union[str, Path]):
    """Loads a Photoluminescence file and generates an interactive HTML Plotly line plot."""
    from defectpl.defectpl import Photoluminescence  # Runtime lazy-import safeguards execution
    
    pl = loadfn(str(filename))
    if not isinstance(pl, Photoluminescence):
        raise TypeError(f"The file {filename} does not contain a valid Photoluminescence object.")

    I_abs = np.abs(pl.intensity)
    I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs
    x_energy = np.arange(len(I_norm)) / float(pl.resolution)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_energy, y=I_norm, mode="lines", line=dict(color="black", width=2)))
    
    fig.update_layout(
        title="Photoluminescence Intensity Spectrum",
        xaxis_title="Photon Energy (eV)",
        yaxis_title="Normalized PL Intensity",
        template="plotly_white",
        width=700,
        height=500
    )
    fig.show()


def plot_interactive_S_omega_Sks_Loc_rat_vs_penergy(filename: Union[str, Path]):
    """Loads a Photoluminescence file and generates a multi-axis interactive visual framework."""
    from defectpl.defectpl import Photoluminescence
    
    pl = loadfn(str(filename))
    if not isinstance(pl, Photoluminescence):
        raise TypeError(f"The file {filename} does not contain a valid Photoluminescence object.")

    freq_mev = np.array(pl.frequencies) * 1000.0
    omega_range = pl.omega_range
    omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
    
    max_f = max(pl.frequencies)
    mask = omega_set <= max_f
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=omega_set[mask] * 1000.0,
            y=np.array(pl.S_omega)[mask] / 1000.0,
            mode="lines",
            line=dict(color="black", width=2),
            name="S(ω) Spectral Density",
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=freq_mev,
            y=pl.Sks,
            mode="markers",
            marker=dict(
                size=7,
                color=pl.localization_ratio,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Localization Ratio", x=1.15)
            ),
            name="Partial HR Factor (S_k)",
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Huang-Rhys Spectral Breakdown Engine",
        xaxis_title="Phonon Energy (meV)",
        template="plotly_white",
        width=850,
        height=500
    )
    fig.update_yaxes(title_text="<b>S(ω) Density</b> (1/meV)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Partial HR Factor</b> (S_k)", secondary_y=True)
    
    fig.show()


def comparepl(
    properties_files: List[Union[str, Path]],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legends: Optional[List[str]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    colors: Optional[List[str]] = None,
    fig_format: str = "pdf",
    figsize: Tuple[float, float] = (3.3, 2.5),
):
    """Loads multiple Photoluminescence data frames to plot comparative isotope pathways."""
    from defectpl.defectpl import Photoluminescence
    
    pl_runs = [loadfn(str(f)) for f in properties_files]
    if legends is None:
        legends = [f"Composition {i+1}" for i in range(len(properties_files))]
        
    fig, ax = plt.subplots(figsize=figsize)
    line_colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    for i, pl in enumerate(pl_runs):
        if not isinstance(pl, Photoluminescence):
            raise TypeError(f"File index {i} does not contain a valid Photoluminescence object.")
            
        resolution = float(pl.resolution)
        x_energy = np.arange(len(pl.intensity)) / resolution
        I_abs = np.abs(pl.intensity)
        I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs
        
        ax.plot(x_energy, I_norm, label=legends[i], color=line_colors[i % len(line_colors)])
        
    ax.set_ylabel("PL Intensity (arb. u.)")
    ax.set_xlabel("Photon Energy (eV)")
    
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
        
    ax.set_yticks([])
    ax.legend(loc="best")
    
    if out_dir:
        out_path = Path(out_dir) / f"compare_pl.{fig_format}"
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)