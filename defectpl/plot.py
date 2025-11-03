# Stores all the plotting functions for the DefectPL package.
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from pathlib import Path
from defectpl.utils import *
from defectpl.io import read_properties
import plotly.graph_objects as go


class Plotter:
    def __init__(self):
        pass

        # Plotting Methods

    def plot_penergy_vs_pmode(
        self,
        frequencies,
        plot=False,
        out_dir="./",
        file_name="penergy_vs_pmode",
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the phonon energy vs phonon mode index.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "penergy_vs_pmode.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * 1000
        mode_i = np.linspace(0, len(freq), len(freq))
        plt.figure(figsize=figsize)
        plt.plot(mode_i, freq)
        plt.xlabel(r"Phonon Mode Index")
        plt.ylabel(r"Phonon Energy (meV)")
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form not in ["png", "pdf", "svg", "jpg", "jpeg"]:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_ipr_vs_penergy(
        self,
        frequencies,
        iprs,
        plot=False,
        out_dir="./",
        file_name="ipr_vs_penergy",
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the IPR vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        iprs: list
            List of IPR values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "ipr_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * 1000
        ipr_data = iprs
        plt.figure(figsize=figsize)
        color = "tab:blue"
        plt.scatter(
            freq, ipr_data, color=color, s=5, linewidths=0.3, edgecolors="k", alpha=0.7
        )
        plt.xlabel(r"Phonon Energy (meV)")
        plt.ylabel(r"IPR")
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_loc_rat_vs_penergy(
        self,
        frequencies,
        localization_ratio,
        plot=False,
        out_dir="./",
        file_name="loc_rat_vs_penergy",
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the localization ratio vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        localization_ratio: list
            List of Localization ratio values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * 1000
        plt.figure(figsize=figsize)
        color = "tab:green"
        plt.scatter(
            freq,
            localization_ratio,
            color="tab:green",
            s=5,
            linewidths=0.3,
            edgecolors="k",
            alpha=0.7,
        )
        plt.xlabel(r"Phonon Energy (meV)")
        plt.ylabel(r"Localization Ratio")
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_qk_vs_penergy(
        self,
        frequencies,
        qks,
        plot=False,
        out_dir="./",
        file_name="qk_vs_penergy",
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the vibrational displacement vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        qks: list
            List of vibrational displacement values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "qk_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * 1000
        q_data = qks
        plt.figure(figsize=figsize)
        color = "tab:blue"
        plt.scatter(
            freq, q_data, color=color, s=5, linewidths=0.3, edgecolors="k", alpha=0.7
        )
        plt.xlabel(r"Phonon Energy (meV)")
        plt.ylabel(r"Vibrational Displacement ($q_k$)")
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_HR_factor_vs_penergy(
        self,
        frequencies,
        Sks,
        plot=False,
        out_dir="./",
        file_name="HR_factor_vs_penergy",
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the partial HR factor vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        Sks: list
            List of partial HR factor values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "HR_factor_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * 1000
        phr_data = Sks
        plt.figure(figsize=figsize)
        color = "tab:blue"
        plt.scatter(
            freq, phr_data, color=color, s=5, linewidths=0.3, edgecolors="k", alpha=0.7
        )
        plt.xlabel(r"Phonon Energy (meV)")
        plt.ylabel(r"Partial HR factor($S_i$)")
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_S_omega_vs_penergy(
        self,
        frequencies,
        S_omega,
        omega_range,
        plot=False,
        out_dir="./",
        file_name="S_omega_vs_penergy",
        max_freq=None,
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """
        Plot the S(omega) vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        if max_freq is None:
            max_freq = max(frequencies)
        plt.figure(figsize=figsize)
        omega_set = np.linspace(omega_range[0], omega_range[1], omega_range[2])
        x = [i for i in omega_set if i <= max_freq]
        Somg = np.array(S_omega[: len(x)]) / 1000
        plt.plot(omega_set[: len(x)] * 1000, Somg, "k")
        plt.ylabel(r"$S(\hbar\omega)(1/meV)$")
        plt.xlabel(r"Phonon energy (meV)")
        plt.xlim(0, max_freq * 1000)
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_S_omega_Sks_vs_penergy(
        self,
        frequencies,
        S_omega,
        omega_range,
        Sks,
        plot=False,
        out_dir="./",
        file_name="S_omega_vs_penergy",
        max_freq=None,
        fig_format="pdf",
        figsize=(6, 4),
    ):
        """
        Plot the S(omega) vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * EV2mEV
        if max_freq is None:
            max_freq = max(frequencies)
        S = Sks
        # Plot S(omega) vs phonon energy
        fig, ax1 = plt.subplots(figsize=figsize)
        color = "tab:green"
        ax1.set_xlabel(r"Phonon energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)(1/meV)$", color=color)
        omega_set = np.linspace(omega_range[0], omega_range[1], omega_range[2])
        x = [i for i in omega_set if i <= max_freq]
        Somg = np.array(S_omega[: len(x)]) / 1000
        ax1.plot(omega_set[: len(x)] * EV2mEV, Somg, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        # ax1.set_ylim(-0.1, 28)

        # Plot Sks vs phonon energy
        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel(r"Partial HR factor", color=color)
        ax2.scatter(
            freq, S, color=color, s=5, linewidths=0.3, edgecolors="k", alpha=0.7
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.xlim(0, max_freq * EV2mEV)
        # ax2.set_ylim(-0.002, 0.38)
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_S_omega_Sks_Loc_rat_vs_penergy(
        self,
        frequencies,
        S_omega,
        omega_range,
        Sks,
        localization_ratio,
        plot=False,
        out_dir="./",
        file_name="S_omega_HRf_loc_rat_vs_penergy",
        max_freq=None,
        pylim=[None, None],
        fig_format="pdf",
        figsize=(6, 4),
    ):
        """
        Plot the S(omega), partial HR factor and localization ratio vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        localization_ratio: list
            List of localization ratio values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_HRf_loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * EV2mEV
        if max_freq is None:
            max_freq = max(frequencies)
        S = Sks
        # Plot S(omega) vs phonon energy
        fig, ax1 = plt.subplots(figsize=figsize)
        color = "tab:blue"
        ax1.set_xlabel(r"Phonon energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)(1/meV)$", color=color)
        omega_set = np.linspace(omega_range[0], omega_range[1], omega_range[2])
        x = [i for i in omega_set if i <= max_freq]
        Somg = np.array(S_omega[: len(x)]) / 1000
        ax1.plot(omega_set[: len(x)] * 1000, Somg, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        # ax1.set_ylim(-0.1, 28)

        # Plot Sks vs phonon energy and localization as color of points
        ax2 = ax1.twinx()
        color = "black"
        cm = plt.get_cmap("cool")
        ax2.set_ylabel(r"Partial HR factor", color=color)
        sc = ax2.scatter(
            freq,
            Sks,
            c=localization_ratio,
            cmap=cm,
            s=10,
            linewidths=0.3,
            edgecolors="k",
            alpha=0.7,
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(pylim[0], pylim[1])

        plt.xlim(0, max_freq * 1000)
        # ax2.set_ylim(-0.002, 0.38)
        cbar = plt.colorbar(sc, pad=0.15)
        cbar.set_label(r"Localization Ratio")  # Add label to the colormap
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_S_omega_Sks_ipr_vs_penergy(
        self,
        frequencies,
        S_omega,
        omega_range,
        Sks,
        iprs,
        plot=False,
        out_dir="./",
        file_name="S_omega_HRf_ipr_vs_penergy",
        max_freq=None,
        fig_format="pdf",
        figsize=(6, 4),
    ):
        """
        Plot the S(omega), partial HR factor and IPR vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        iprs: list
            List of IPR values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_HRf_loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        freq = frequencies * EV2mEV
        if max_freq is None:
            max_freq = max(frequencies)
        S = Sks
        # Plot S(omega) vs phonon energy
        fig, ax1 = plt.subplots(figsize=figsize)
        color = "tab:blue"
        ax1.set_xlabel(r"Phonon energy (meV)")
        ax1.set_ylabel(r"$S(\hbar\omega)(1/meV)$", color=color)
        omega_set = np.linspace(omega_range[0], omega_range[1], omega_range[2])
        x = [i for i in omega_set if i <= max_freq]
        Somg = np.array(S_omega[: len(x)]) / 1000
        ax1.plot(omega_set[: len(x)] * 1000, Somg, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        # ax1.set_ylim(-0.1, 28)

        # Plot Sks vs phonon energy and localization as color of points
        ax2 = ax1.twinx()
        color = "black"
        cm = plt.get_cmap("cool")
        ax2.set_ylabel(r"Partial HR factor", color=color)
        sc = ax2.scatter(
            freq, Sks, c=iprs, cmap=cm, s=10, linewidths=0.3, edgecolors="k", alpha=0.7
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.xlim(0, max_freq * 1000)
        # ax2.set_ylim(-0.002, 0.38)
        cbar = plt.colorbar(sc, pad=0.15)
        cbar.set_label(r"Inverse Participation Ratio")  # Add label to the colormap
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()

    def plot_intensity_vs_penergy(
        self,
        frequencies,
        I,
        resolution,
        xlim,
        plot=False,
        out_dir="./",
        file_name="intensity_vs_penergy",
        iylim=None,
        fig_format="pdf",
        figsize=(4, 4),
    ):
        """Plot the intensity vs phonon energy.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        I: np.array of complex values
            List of intensity values for each phonon mode.
        resolution: float
            Resolution of the time-domain signal.
        xlim: list
            Range of phonon energy values. [Start, End]. Unit meV.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "intensity_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.
        """
        file_name = f"{file_name}.{fig_format}"
        out_path = Path(out_dir) / file_name
        plt.figure(figsize=figsize)
        I_abs = I.__abs__()
        I_abs = I_abs / np.max(I_abs)
        plt.plot(I_abs, "k")
        # plt.ylabel(r"$I(\hbar\omega)$")
        plt.ylabel(r"PL intensity")
        plt.xlabel(r"Photon energy (eV)")
        plt.xlim(xlim[0], xlim[1])
        x_values, labels = plt.xticks()
        labels = [float(x) / resolution for x in x_values]
        plt.xticks(x_values, labels)
        if iylim:
            plt.ylim(iylim[0], iylim[1])
        plt.yticks([])
        if plot:
            plt.show()
        else:
            form = file_name.split(".")[-1]
            if form:
                form = form.lower()
            else:
                form = "pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
            plt.close()


def plot_interactive_intensity(filename):
    """
    Plot the interactive intensity plot.
    """
    properties = read_properties(filename)
    I = properties["I"]
    resolution = properties["resolution"]
    I_abs = I.__abs__()
    x_values = list(range(len(I_abs)))
    x_values = np.array(x_values) / resolution
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x_values, y=I_abs, mode="lines", line=dict(color="black"))
    )

    fig.show()


def plot_interactive_S_omega_Sks_Loc_rat_vs_penergy(filename):
    """
    Plot the interactive S(omega), partial HR factor and localization ratio vs phonon energy.
    """
    properties = read_properties(filename)
    freq = np.array(properties["frequencies"]) * EV2mEV
    S = properties["Sks"]
    S_omega = properties["S_omega"]
    loc_rat = properties["localization_ratio"]
    max_freq = max(freq)
    omega_set = np.linspace(
        properties["omega_range"][0],
        properties["omega_range"][1],
        properties["omega_range"][2],
    )
    x = [i for i in omega_set if i <= max_freq]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=omega_set[: len(x)] * 1000,
            y=S_omega[: len(x)],
            mode="lines",
            line=dict(color="black"),
            name="S(omega)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=S,
            mode="markers",
            marker=dict(color="blue"),
            name="Partial HR factor",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=loc_rat,
            mode="markers",
            marker=dict(color="green"),
            name="Localization Ratio",
        )
    )

    fig.show()


def comparepl(
    properties_files,
    xlim=None,
    ylim=None,
    legends=None,
    out_dir=None,
    colors=None,
    fig_format="pdf",
    figsize=(4, 4),
):
    """
    Compare the PL of different isotopic compositions.
    """

    properties = []
    for filename in properties_files:
        properties.append(read_properties(filename))
    I = [prop["I"] for prop in properties]
    if xlim is None:
        xlim = properties[0]["omega_range"]
    else:
        xlim = xlim
    resolution = properties[0]["resolution"]

    if legends is None:
        legends = [f"Composition {i+1}" for i in range(len(properties_files))]
    # Create a matplotlib figure for each intensity
    fig, ax = plt.subplots(figsize=figsize)
    for i, intensity in enumerate(I):
        I_abs = intensity.__abs__()
        I_abs = I_abs / np.max(I_abs)
        if colors:
            ax.plot(I_abs, label=legends[i], color=colors[i])
        else:
            ax.plot(I_abs, label=legends[i])
    ax.set_ylabel(r"$I(\hbar\omega)$")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    x_values, labels = plt.xticks()
    labels = [float(x) / resolution for x in x_values]
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_yticks([], [])
    ax.legend(loc=0)
    if out_dir:
        plt.savefig(
            Path(out_dir) / f"compare_pl.{fig_format}", dpi=300, bbox_inches="tight"
        )
    else:
        plt.show()
