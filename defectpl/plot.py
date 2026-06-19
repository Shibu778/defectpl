# -*- coding: utf-8 -*-
"""
Stores publication-grade plotting functions for the defectpl package.
Author : Shibu Meher
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monty.serialization import loadfn


# Load custom publication style with a robust procedural fallback layout
style_file = Path(__file__).parent / "defectpl.mplstyle"
if style_file.exists():
    style.use(str(style_file))
    import shutil as _shutil

    if plt.rcParams.get("text.usetex") and _shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


class Plotter:
    """
    Publication-quality static figure generator for DefectPL outputs.

    Each method produces a single matplotlib figure and either displays it
    interactively (``plot=True``) or saves it at 600 dpi to *out_dir*
    (``plot=False``, the default used by :meth:`~defectpl.defectpl.Photoluminescence.generate_plots`).

    All energy inputs follow the internal convention of **eV**; axis labels
    are converted to meV where appropriate for readability.

    Methods
    -------
    plot_penergy_vs_pmode
        Phonon energy vs mode index (dispersionless Γ-point spectrum).
    plot_ipr_vs_penergy
        Phonon IPR vs phonon energy.
    plot_loc_rat_vs_penergy
        Localization ratio β_k = N·IPR_k vs phonon energy.
    plot_qk_vs_penergy
        Mode-projected displacement q_k vs phonon energy.
    plot_HR_factor_vs_penergy
        Partial Huang–Rhys factors S_k vs phonon energy.
    plot_S_omega_vs_penergy
        Continuous spectral density S(ω) vs phonon energy.
    plot_S_omega_Sks_vs_penergy
        S(ω) and S_k overlaid on dual y-axes.
    plot_S_omega_Sks_Loc_rat_vs_penergy
        S(ω) + S_k scatter coloured by localization ratio.
    plot_S_omega_Sks_ipr_vs_penergy
        S(ω) + S_k scatter coloured by IPR (traditional convention).
    plot_ipr_alkauskas_vs_penergy
        Phonon IPR (Alkauskas eq. 12 convention) vs phonon energy.
    plot_S_omega_Sks_ipr_alkauskas_vs_penergy
        S(ω) + S_k scatter coloured by Alkauskas IPR.
    plot_nk_vs_penergy
        Bose-Einstein phonon occupation n̄_k(T) vs phonon energy.
    plot_C_omega_vs_penergy
        Thermal spectral density C(ℏω, T) vs phonon energy.
    plot_intensity_vs_penergy
        Normalised PL intensity spectrum vs photon energy.
    plot_absorption_vs_penergy
        Normalised absorption spectrum vs photon energy.
    plot_pl_absorption_vs_penergy
        PL and absorption overlaid on the same axes.
    """

    def __init__(self):
        pass

    def _save_or_show(
        self,
        fig,
        out_dir: Union[str, Path],
        file_name: str,
        fig_format: str,
        plot: bool,
    ):
        """Save figure to *out_dir*/<file_name>.<fmt> at 600 dpi, or show interactively."""
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
        """
        Plot phonon energy (meV) vs mode index.

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        plot : bool, optional
            Show interactively instead of saving.  Default False.
        out_dir : str or Path, optional
            Output directory.  Default ``"./"``
        file_name : str, optional
            Base file name (no extension).  Default ``"penergy_vs_pmode"``.
        fig_format : str, optional
            ``"pdf"``, ``"png"``, ``"svg"``, or ``"jpg"``.  Default ``"pdf"``.
        figsize : tuple of float, optional
            Figure size in inches.  Default ``(3.3, 2.5)``.
        """
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
        """
        Scatter plot of phonon IPR vs phonon energy (meV).

        IPR = Σp²/(Σp)² ranges from 1/N (fully delocalized) to 1 (fully localized).

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        iprs : numpy.ndarray, shape (nmodes,)
            Inverse participation ratios per mode.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        """
        Scatter plot of localization ratio β_k = N·IPR_k vs phonon energy (meV).

        β = 1 means perfectly delocalized; β = N means fully localized on one atom.

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        localization_ratio : numpy.ndarray, shape (nmodes,)
            Localization ratio per mode.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        """
        Scatter plot of mode-projected displacement q_k vs phonon energy (meV).

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        qks : numpy.ndarray, shape (nmodes,)
            Mode configurational displacements in amu^(1/2)·Å.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        """
        Scatter plot of partial Huang–Rhys factors S_k vs phonon energy (meV).

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        Sks : numpy.ndarray, shape (nmodes,)
            Partial (mode-resolved) Huang–Rhys factors.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        """
        Line plot of the continuous Huang–Rhys spectral density S(ω) in 1/meV vs energy (meV).

        S(ω) is the phonon sideband weight per unit energy; its integral equals
        the total Huang–Rhys factor S.

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV** (used to set x-axis upper limit).
        S_omega : array-like, shape (n_grid,)
            Spectral density in **eV^(-1)**.
        omega_range : list [ω_min, ω_max, n_points]
            Energy grid parameters in **eV** matching the S(ω) array.
        max_freq : float, optional
            Upper frequency cut-off for the plot in **eV**.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        """
        Dual-axis plot of S(ω) (left axis) and partial S_k scatter (right axis).

        Parameters
        ----------
        frequencies, S_omega, omega_range
            See :meth:`plot_S_omega_vs_penergy`.
        Sks : numpy.ndarray, shape (nmodes,)
            Partial Huang–Rhys factors.
        max_freq : float, optional
            Upper frequency cut-off in **eV**.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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
        ax2.scatter(
            np.array(frequencies) * 1000.0, Sks, color=c2, edgecolor="black", alpha=0.85
        )
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
        """
        S(ω) line + S_k scatter coloured by localization ratio β_k.

        Identifies phonon modes that simultaneously carry large HR weight *and*
        are localized near the defect.

        Parameters
        ----------
        frequencies, S_omega, omega_range
            See :meth:`plot_S_omega_vs_penergy`.
        Sks : numpy.ndarray, shape (nmodes,)
            Partial Huang–Rhys factors (right y-axis).
        localization_ratio : numpy.ndarray, shape (nmodes,)
            Colour-code values β_k = N·IPR_k.
        pylim : list of [float or None, float or None], optional
            ``[ymin, ymax]`` for the right (S_k) axis.
        cmap : str, optional
            Matplotlib colormap name.  Default ``"viridis"``.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        # FIX: Explicitly set layout="constrained" to cleanly handle twinx + colorbar spacing
        fig, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))

        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff

        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color="black")
        ax1.plot(
            omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color="black"
        )
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color="black")

        sc = ax2.scatter(
            np.array(frequencies) * 1000.0,
            Sks,
            c=localization_ratio,
            cmap=plt.get_cmap(cmap),
            edgecolor="black",
            alpha=0.85,
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
        """
        S(ω) line + S_k scatter coloured by phonon IPR.

        Parameters
        ----------
        frequencies, S_omega, omega_range
            See :meth:`plot_S_omega_vs_penergy`.
        Sks : numpy.ndarray, shape (nmodes,)
            Partial Huang–Rhys factors (right y-axis).
        iprs : numpy.ndarray, shape (nmodes,)
            Phonon inverse participation ratios used as colour code.
        cmap : str, optional
            Matplotlib colormap name.  Default ``"viridis"``.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        # FIX: Explicitly set layout="constrained" to cleanly handle twinx + colorbar spacing
        fig, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))

        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff

        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color="black")
        ax1.plot(
            omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color="black"
        )
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color="black")

        sc = ax2.scatter(
            np.array(frequencies) * 1000.0,
            Sks,
            c=iprs,
            cmap=plt.get_cmap(cmap),
            edgecolor="black",
            alpha=0.85,
        )
        ax2.tick_params(axis="y", labelcolor="black")
        ax1.set_xlim(0, cutoff * 1000.0)

        # When using layout="constrained", we can target ax2 directly with an explicit pad
        cbar = fig.colorbar(sc, ax=ax2, pad=0.12)
        cbar.set_label("Inverse Participation Ratio")

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_ipr_alkauskas_vs_penergy(
        self,
        frequencies: np.ndarray,
        iprs_alkauskas: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "ipr_alkauskas_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """
        Scatter plot of phonon IPR (Alkauskas eq. 12 convention) vs phonon energy (meV).

        IPR_k = (Σp)²/Σp² ranges from 1 (fully localized) to N (fully delocalized).
        This is the reciprocal of the traditional IPR plotted by :meth:`plot_ipr_vs_penergy`.

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        iprs_alkauskas : numpy.ndarray, shape (nmodes,)
            Alkauskas-convention IPR per mode (Alkauskas 2014, eq. 12).
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0

        ax.scatter(freq_mev, iprs_alkauskas, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(r"$\mathrm{IPR}_k$ (Alkauskas)")

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_S_omega_Sks_ipr_alkauskas_vs_penergy(
        self,
        frequencies: np.ndarray,
        S_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        Sks: np.ndarray,
        iprs_alkauskas: np.ndarray,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "S_omega_HRf_ipr_alkauskas_vs_penergy",
        max_freq: Optional[float] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (4.2, 2.5),
        cmap: str = "viridis",
    ):
        """
        S(ω) line + S_k scatter coloured by Alkauskas-convention phonon IPR.

        Modes with low IPR_k (close to 1) are localized; modes with high IPR_k
        (close to N) are bulk-like delocalized.

        Parameters
        ----------
        frequencies, S_omega, omega_range
            See :meth:`plot_S_omega_vs_penergy`.
        Sks : numpy.ndarray, shape (nmodes,)
            Partial Huang–Rhys factors (right y-axis).
        iprs_alkauskas : numpy.ndarray, shape (nmodes,)
            Alkauskas-convention IPR per mode used as colour code.
        cmap : str, optional
            Matplotlib colormap name.  Default ``"viridis"``.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))

        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff

        ax1.set_xlabel("Phonon Energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)$ ($1/\mathrm{meV}$)", color="black")
        ax1.plot(
            omega_set[mask] * 1000.0, np.array(S_omega)[mask] / 1000.0, color="black"
        )
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Partial HR Factor ($S_k$)", color="black")

        sc = ax2.scatter(
            np.array(frequencies) * 1000.0,
            Sks,
            c=iprs_alkauskas,
            cmap=plt.get_cmap(cmap),
            edgecolor="black",
            alpha=0.85,
        )
        ax2.tick_params(axis="y", labelcolor="black")
        ax1.set_xlim(0, cutoff * 1000.0)

        cbar = fig.colorbar(sc, ax=ax2, pad=0.12)
        cbar.set_label(r"$\mathrm{IPR}_k$ (Alkauskas)")

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
        """
        Line plot of the normalised PL intensity spectrum vs photon energy (eV).

        Parameters
        ----------
        frequencies : numpy.ndarray
            Phonon frequencies in eV (unused directly; kept for API consistency).
        I : numpy.ndarray
            Complex or real intensity array from :func:`~defectpl.utils.calc_Spectrum_Intensity`.
            Absolute value is taken and the result is normalised to its maximum.
        resolution : int
            Number of grid points per eV; sets the energy-axis scale.
        xlim : tuple of float
            ``(x_min, x_max)`` for the photon energy axis in eV.
        iylim : tuple of float, optional
            ``(y_min, y_max)`` for the intensity axis.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
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

    def plot_nk_vs_penergy(
        self,
        frequencies: np.ndarray,
        nks: np.ndarray,
        temperature: float = 0.0,
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "nk_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """
        Scatter plot of Bose-Einstein phonon occupation n̄_k(T) vs phonon energy (meV).

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        nks : numpy.ndarray, shape (nmodes,)
            Bose-Einstein occupation numbers per mode.
        temperature : float, optional
            Temperature in K, displayed in the axis label.  Default 0.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax = plt.subplots(figsize=figsize)
        freq_mev = np.array(frequencies) * 1000.0

        ax.scatter(freq_mev, nks, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(rf"$\bar{{n}}_k$ ({temperature:.0f} K)")

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_C_omega_vs_penergy(
        self,
        frequencies: np.ndarray,
        C_omega: Union[list, np.ndarray],
        omega_range: List[Union[int, float]],
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "C_omega_vs_penergy",
        max_freq: Optional[float] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """
        Line plot of the thermal spectral density C(ℏω, T) in 1/meV vs energy (meV).

        C(ω, T) = Σ_k n̄_k(T) S_k δ(ℏω − ℏω_k) broadened with Gaussians.
        At T = 0 this plot is a flat zero line.

        Parameters
        ----------
        frequencies : numpy.ndarray, shape (nmodes,)
            Phonon frequencies in **eV**.
        C_omega : array-like, shape (n_grid,)
            Thermal spectral density in **eV^(-1)**.
        omega_range : list [ω_min, ω_max, n_points]
            Energy grid parameters in **eV**.
        max_freq : float, optional
            Upper frequency cut-off in **eV**.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax = plt.subplots(figsize=figsize)
        omega_set = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))

        cutoff = max_freq if max_freq is not None else float(max(frequencies))
        mask = omega_set <= cutoff

        C_omega_mev = np.array(C_omega)[mask] / 1000.0
        energies_mev = omega_set[mask] * 1000.0

        ax.plot(energies_mev, C_omega_mev, color="black")
        ax.set_xlabel("Phonon Energy (meV)")
        ax.set_ylabel(r"$C(\hbar\omega,T)$ ($1/\mathrm{meV}$)")
        ax.set_xlim(0, cutoff * 1000.0)

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_absorption_vs_penergy(
        self,
        frequencies: np.ndarray,
        absorption: np.ndarray,
        resolution: int,
        xlim: Tuple[float, float],
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "absorption_vs_penergy",
        iylim: Optional[Tuple[float, float]] = None,
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """
        Line plot of the normalised absorption spectrum vs photon energy (eV).

        The sideband lies on the **high-energy side** of the ZPL (phonon
        absorption raises the photon energy), opposite to PL emission.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Phonon frequencies in eV (unused directly; kept for API consistency).
        absorption : numpy.ndarray
            Complex or real absorption array from
            :func:`~defectpl.utils.calc_Absorption_Intensity`.
            Absolute value is taken and the result is normalised.
        resolution : int
            Number of grid points per eV.
        xlim : tuple of float
            ``(x_min, x_max)`` for the photon energy axis in eV.
        iylim : tuple of float, optional
            ``(y_min, y_max)`` for the intensity axis.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax = plt.subplots(figsize=figsize)

        x_energy_ev = np.arange(len(absorption)) / float(resolution)
        I_abs = np.abs(absorption)
        I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs

        ax.plot(x_energy_ev, I_norm, color="black")
        ax.set_ylabel("Absorption (arb. u.)")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_xlim(xlim[0], xlim[1])

        if iylim:
            ax.set_ylim(iylim[0], iylim[1])
        ax.set_yticks([])

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)

    def plot_pl_absorption_vs_penergy(
        self,
        frequencies: np.ndarray,
        intensity: np.ndarray,
        absorption: np.ndarray,
        resolution: int,
        xlim: Tuple[float, float],
        plot: bool = False,
        out_dir: Union[str, Path] = "./",
        file_name: str = "pl_absorption_vs_penergy",
        fig_format: str = "pdf",
        figsize: Tuple[float, float] = (3.3, 2.5),
    ):
        """
        Overlay of normalised PL (solid) and absorption (dashed) spectra.

        Displays the Stokes shift between the PL emission peak and the
        absorption onset on a shared photon-energy axis.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Phonon frequencies in eV (unused directly; kept for API consistency).
        intensity : numpy.ndarray
            PL intensity array from :func:`~defectpl.utils.calc_Spectrum_Intensity`.
        absorption : numpy.ndarray
            Absorption array from :func:`~defectpl.utils.calc_Absorption_Intensity`.
        resolution : int
            Number of grid points per eV.
        xlim : tuple of float
            ``(x_min, x_max)`` for the photon energy axis in eV.
        plot, out_dir, file_name, fig_format, figsize
            See :meth:`plot_penergy_vs_pmode`.
        """
        fig, ax = plt.subplots(figsize=figsize)

        x_ev = np.arange(len(intensity)) / float(resolution)

        I_pl = np.abs(intensity)
        I_pl = I_pl / np.max(I_pl) if np.max(I_pl) > 0 else I_pl

        I_abs = np.abs(absorption)
        I_abs = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax.plot(x_ev, I_pl, color=colors[0], label="PL")
        ax.plot(x_ev, I_abs, color=colors[1], linestyle="--", label="Absorption")

        ax.set_ylabel("Intensity (arb. u.)")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_yticks([])
        ax.legend(loc="best")

        self._save_or_show(fig, out_dir, file_name, fig_format, plot)


# =====================================================================
# Standalone Interactive Plotly & Comparison Utilities
# =====================================================================


def plot_interactive_intensity(filename: Union[str, Path]):
    """
    Open an interactive Plotly PL intensity spectrum from a serialised Photoluminescence JSON.

    Parameters
    ----------
    filename : str or Path
        Path to a JSON file written by :meth:`~defectpl.defectpl.Photoluminescence.as_dict`.

    Raises
    ------
    TypeError
        If the file does not contain a :class:`~defectpl.defectpl.Photoluminescence` object.
    """
    from defectpl.defectpl import (
        Photoluminescence,
    )  # Runtime lazy-import safeguards execution

    pl = loadfn(str(filename))
    if not isinstance(pl, Photoluminescence):
        raise TypeError(
            f"The file {filename} does not contain a valid Photoluminescence object."
        )

    I_abs = np.abs(pl.intensity)
    I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs
    x_energy = np.arange(len(I_norm)) / float(pl.resolution)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_energy, y=I_norm, mode="lines", line=dict(color="black", width=2)
        )
    )

    fig.update_layout(
        title="Photoluminescence Intensity Spectrum",
        xaxis_title="Photon Energy (eV)",
        yaxis_title="Normalized PL Intensity",
        template="plotly_white",
        width=700,
        height=500,
    )
    fig.show()


def plot_interactive_S_omega_Sks_Loc_rat_vs_penergy(filename: Union[str, Path]):
    """
    Open an interactive Plotly figure showing S(ω), S_k, and localization ratio.

    Loads a serialised :class:`~defectpl.defectpl.Photoluminescence` object and
    renders a dual-y-axis Plotly figure with S(ω) as a line and S_k
    as a scatter coloured by the localization ratio.

    Parameters
    ----------
    filename : str or Path
        Path to the serialised Photoluminescence JSON.

    Raises
    ------
    TypeError
        If the file does not contain a valid Photoluminescence object.
    """
    from defectpl.defectpl import Photoluminescence

    pl = loadfn(str(filename))
    if not isinstance(pl, Photoluminescence):
        raise TypeError(
            f"The file {filename} does not contain a valid Photoluminescence object."
        )

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
        secondary_y=False,
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
                colorbar=dict(title="Localization Ratio", x=1.15),
            ),
            name="Partial HR Factor (S_k)",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Huang-Rhys Spectral Breakdown Engine",
        xaxis_title="Phonon Energy (meV)",
        template="plotly_white",
        width=850,
        height=500,
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
    """
    Overlay normalised PL spectra from multiple Photoluminescence JSON files.

    Useful for comparing spectra computed with different isotope masses,
    temperatures, or structural parameters.

    Parameters
    ----------
    properties_files : list of str or Path
        Paths to serialised Photoluminescence JSON files.
    xlim : tuple of float, optional
        ``(x_min, x_max)`` for the photon energy axis in eV.
    ylim : tuple of float, optional
        ``(y_min, y_max)`` for the intensity axis.
    legends : list of str, optional
        Legend labels for each spectrum.  Defaults to ``"Composition 1"``, etc.
    out_dir : str or Path, optional
        If given, save the figure there instead of showing interactively.
    colors : list of str, optional
        Matplotlib color strings for each spectrum.
    fig_format : str, optional
        Output format.  Default ``"pdf"``.
    figsize : tuple of float, optional
        Figure size in inches.  Default ``(3.3, 2.5)``.
    """
    from defectpl.defectpl import Photoluminescence

    pl_runs = [loadfn(str(f)) for f in properties_files]
    if legends is None:
        legends = [f"Composition {i + 1}" for i in range(len(properties_files))]

    fig, ax = plt.subplots(figsize=figsize)
    line_colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, pl in enumerate(pl_runs):
        if not isinstance(pl, Photoluminescence):
            raise TypeError(
                f"File index {i} does not contain a valid Photoluminescence object."
            )

        resolution = float(pl.resolution)
        x_energy = np.arange(len(pl.intensity)) / resolution
        I_abs = np.abs(pl.intensity)
        I_norm = I_abs / np.max(I_abs) if np.max(I_abs) > 0 else I_abs

        ax.plot(
            x_energy, I_norm, label=legends[i], color=line_colors[i % len(line_colors)]
        )

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
