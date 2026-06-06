# -*- coding: utf-8 -*-
"""
Defect Optical Properties Engine (DefectPL) core module.
Authors: Shibu Meher, Manoj Dey
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from monty.json import MSONable
from pymatgen.core import Structure

from defectpl.constants import AMU2KG, ANG2M, EV2J, HBAR_EVS
from defectpl.plot import Plotter
import defectpl.utils as utils
from defectpl.vasp_wrapper import calc_delta_Q, get_q_from_structure


@dataclass
class Photoluminescence(MSONable):
    """
    Standard core engine computing defect PL state dynamics.

    Inherits from MSONable for JSON serialization.

    When dR is provided, qks is computed using calc_qks function.
    However, when dF is provided, qks is computed using calc_qks_force_mode.
    """

    # 1. Mandatory Core Inputs (No Default Values Allowed First)
    frequencies: np.ndarray  # (nmodes,) Mode phonon energies in eV
    eigenvectors: np.ndarray  # (nmodes, natoms, 3) Displacement matrix vectors
    masses: np.ndarray  # (natoms,) Atomic mass structural log in AMU
    EZPL: float  # Zero phonon line energy in eV

    # 2. Runtime Optional Parameters & Settings (Defaults Grouped Last)
    dR: Optional[np.ndarray] = (
        None  # (natoms, 3) Atomic structural shift (Excited - Ground) in Å
    )
    dF: Optional[np.ndarray] = (
        None  # (natoms, 3) Atomic force shift (Excited - Ground) in eV/Å
    )
    resolution: int = 1000  # Density step intervals per 1 eV boundary limit
    max_energy: float = 5.0  # Range tracking upper caps conditions in eV
    sigma: float = 6e-3  # Continuous broadening profile used in Gaussian calculations
    gamma: float = 2.0  # Homogeneous/inhomogeneous ZPL broadening factor

    # Dependent calculated properties stored dynamically downstream
    natoms: int = field(init=False)
    delR: float = field(init=False, default=None)
    delQ: float = field(init=False, default=None)
    qks: np.ndarray = field(init=False, default=None)
    Sks: np.ndarray = field(init=False, default=None)
    HR_factor: float = field(init=False, default=None)
    DW_factor: float = field(init=False, default=None)
    iprs: np.ndarray = field(init=False, default=None)
    localization_ratio: np.ndarray = field(init=False, default=None)
    omega_range: List[Union[float, int]] = field(init=False, default=None)
    S_omega: np.ndarray = field(init=False, default=None)
    Sts: np.ndarray = field(init=False, default=None)
    Gts: np.ndarray = field(init=False, default=None)
    A_line: np.ndarray = field(init=False, default=None)
    intensity: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.frequencies = np.asarray(self.frequencies)
        self.eigenvectors = np.asarray(self.eigenvectors)
        self.masses = np.asarray(self.masses)
        if self.dR is not None:
            self.dR = np.asarray(self.dR)
        if self.dF is not None:
            self.dF = np.asarray(self.dF)

        self.natoms = len(self.masses)
        self.omega_range = [
            0.0,
            self.max_energy,
            int(self.max_energy * self.resolution),
        ]
        self.compute_properties()

    def compute_properties(self):
        """Executes the calculation pipeline using detached utility modules."""
        self.delR = utils.calc_delR(self.dR) if self.dR is not None else 0.0
        self.delQ = (
            utils.calc_delQ(self.masses, self.dR) if self.dR is not None else 0.0
        )

        if self.dF is not None and np.any(self.dF):
            self.qks = utils.calc_qks_force_mode(
                self.masses, self.dF, self.eigenvectors, self.frequencies
            )
        elif self.dR is not None and np.any(self.dR):
            self.qks = utils.calc_qks(self.masses, self.dR, self.eigenvectors)
        else:
            raise ValueError(
                "Either dR or dF must be provided and non-zero to compute qks."
            )

        self.Sks = utils.calc_Sks(self.qks, self.frequencies)
        self.HR_factor = float(np.sum(self.Sks))
        self.DW_factor = float(np.exp(-self.HR_factor))

        self.iprs = utils.calc_IPR(self.eigenvectors)
        self.localization_ratio = self.natoms / self.iprs

        self.S_omega = utils.calc_S_omega(
            self.frequencies, self.Sks, self.omega_range, self.sigma
        )
        self.Sts = utils.calc_St(self.S_omega)
        self.Gts = utils.calc_Gts(self.Sts, self.HR_factor, self.gamma, self.resolution)
        self.A_line, self.intensity = utils.calc_Spectrum_Intensity(
            self.Gts, self.EZPL, self.resolution
        )

    def as_dict(self) -> dict:
        """
        Exports core parameters and safe real-valued arrays, leaving
        complex-valued or massive spectral tracking lines out of the payload.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            # Core Inputs
            "frequencies": self.frequencies.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "masses": self.masses.tolist(),
            "dR": self.dR.tolist() if self.dR is not None else None,
            "dF": self.dF.tolist() if self.dF is not None else None,
            "EZPL": self.EZPL,
            "gamma": self.gamma,
            "resolution": self.resolution,
            "max_energy": self.max_energy,
            "sigma": self.sigma,
            # Safe Real-Valued Computed Properties
            "natoms": self.natoms,
            "delR": float(self.delR) if hasattr(self.delR, "__float__") else self.delR,
            "delQ": float(self.delQ) if hasattr(self.delQ, "__float__") else self.delQ,
            "qks": self.qks.tolist() if self.qks is not None else None,
            "Sks": self.Sks.tolist() if self.Sks is not None else None,
            "HR_factor": (
                float(self.HR_factor)
                if hasattr(self.HR_factor, "__float__")
                else self.HR_factor
            ),
            "DW_factor": (
                float(self.DW_factor)
                if hasattr(self.DW_factor, "__float__")
                else self.DW_factor
            ),
            "iprs": self.iprs.tolist() if self.iprs is not None else None,
            "localization_ratio": (
                self.localization_ratio.tolist()
                if self.localization_ratio is not None
                else None
            ),
            "omega_range": self.omega_range,
            "S_omega": self.S_omega.tolist() if self.S_omega is not None else None,
            # Explicitly drop complex/spectral properties to avoid serialization bugs
            "Sts": None,
            "Gts": None,
            "A_line": None,
            "intensity": None,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """
        Reconstructs the object by reading core inputs and real values,
        then automatically recomputes missing complex and spectrum properties.
        """
        # Create an uninitialized instance to handle assignment without tripping standard __post_init__ loops
        obj = cls.__new__(cls)

        # Load Core Inputs
        obj.frequencies = np.array(d["frequencies"])
        obj.eigenvectors = np.array(d["eigenvectors"])
        obj.masses = np.array(d["masses"])
        obj.dR = np.array(d["dR"]) if d.get("dR") is not None else None
        obj.dF = np.array(d["dF"]) if d.get("dF") is not None else None
        obj.EZPL = d["EZPL"]
        obj.gamma = d["gamma"]
        obj.resolution = d.get("resolution", 1000)
        obj.max_energy = d.get("max_energy", 5.0)
        obj.sigma = d.get("sigma", 6e-3)

        # Load Stored Real-Valued Properties
        obj.natoms = d.get("natoms", len(obj.masses))
        obj.delR = d.get("delR")
        obj.delQ = d.get("delQ")
        obj.qks = np.array(d["qks"]) if d.get("qks") is not None else None
        obj.Sks = np.array(d["Sks"]) if d.get("Sks") is not None else None
        obj.HR_factor = d.get("HR_factor")
        obj.DW_factor = d.get("DW_factor")
        obj.iprs = np.array(d["iprs"]) if d.get("iprs") is not None else None
        obj.localization_ratio = (
            np.array(d["localization_ratio"])
            if d.get("localization_ratio") is not None
            else None
        )
        obj.omega_range = d.get(
            "omega_range", [0.0, obj.max_energy, int(obj.max_energy * obj.resolution)]
        )
        obj.S_omega = np.array(d["S_omega"]) if d.get("S_omega") is not None else None

        # Set up placeholders for the missing complex/spectral parameters
        obj.Sts = None
        obj.Gts = None
        obj.A_line = None
        obj.intensity = None

        # Fallback Trigger: Recompute the complex-dependent parts of the pipeline on-the-fly
        if obj.intensity is None:
            # Recompute the remaining downstream properties (Sts, Gts, A_line, intensity)
            obj.Sts = utils.calc_St(obj.S_omega)
            obj.Gts = utils.calc_Gts(obj.Sts, obj.HR_factor, obj.gamma, obj.resolution)
            obj.A_line, obj.intensity = utils.calc_Spectrum_Intensity(
                obj.Gts, obj.EZPL, obj.resolution
            )

        return obj

    @classmethod
    def from_dict_expensive(cls, d: dict):
        """
        Reconstructs the object cleanly by extracting ONLY the required primary
        inputs. Dependent arrays are automatically recalculated via __post_init__.
        """
        return cls(
            frequencies=np.array(d["frequencies"]),
            eigenvectors=np.array(d["eigenvectors"]),
            masses=np.array(d["masses"]),
            dR=np.array(d["dR"]) if d.get("dR") is not None else None,
            dF=np.array(d["dF"]) if d.get("dF") is not None else None,
            EZPL=d["EZPL"],
            gamma=d["gamma"],
            resolution=d.get("resolution", 1000),
            max_energy=d.get("max_energy", 5.0),
            sigma=d.get("sigma", 6e-3),
        )

    def generate_plots(
        self,
        out_dir: Union[str, Path],
        max_freq: Optional[float] = None,
        iylim=None,
        fig_format="pdf",
    ):
        """Passes computed state parameters downstream into the external Plotter engine."""
        plotter = Plotter()
        iplot_xlim = (max(0.0, self.EZPL - 2.0), self.EZPL + 1.0)
        freq_limit = (max_freq / 1000.0) if max_freq else None

        plotter.plot_penergy_vs_pmode(
            frequencies=self.frequencies,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_ipr_vs_penergy(
            self.frequencies,
            self.iprs,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_loc_rat_vs_penergy(
            self.frequencies,
            self.localization_ratio,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_qk_vs_penergy(
            self.frequencies,
            self.qks,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_HR_factor_vs_penergy(
            self.frequencies,
            self.Sks,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )

        plotter.plot_S_omega_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_S_omega_Sks_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_S_omega_Sks_Loc_rat_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.localization_ratio,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_S_omega_Sks_ipr_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.iprs,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )

        plotter.plot_intensity_vs_penergy(
            frequencies=self.frequencies,
            I=self.intensity,
            resolution=self.resolution,
            xlim=iplot_xlim,
            plot=False,
            out_dir=out_dir,
            iylim=iylim,
            fig_format=fig_format,
        )
        print("All static visualization plots generated successfully.")


@dataclass
class VibrationalSpectra1D(MSONable):
    """
    1D Harmonic Oscillator model for computing the vibrational lineshape of luminescence bands.
    """

    EZPL: float
    w1_meV: float
    w2_meV: float
    DQ: float
    T: float
    E0: float
    dE: float
    M: int

    NN1: int = 22
    NN2: int = 52

    overlap_matrix: np.ndarray = field(init=False, repr=False)
    overlap_data: Dict[str, List[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    spectral_data: Dict[str, List[float]] = field(
        default_factory=dict, init=False, repr=False
    )

    # Coherent SI Conversion Factors derived directly from package constants
    K2EV: float = 8.617333262e-5
    FACTOR: float = field(init=False)

    def __post_init__(self):
        self.M = int(self.M)
        self.w1 = self.w1_meV / 1000.0
        self.w2 = self.w2_meV / 1000.0

        # Unified mass-weighted conversion factor substitution: sqrt(AMU2KG)*ANG2M / HBAR_J_S
        hbar_j_s = HBAR_EVS * EV2J
        self.FACTOR = np.sqrt(AMU2KG) * ANG2M / hbar_j_s  # Yields ~15.46485

        self.sigma = 0.70 * self.w2
        self.TE = self.T * self.K2EV
        self.w = (
            self.w1 * self.w2 / (self.w1 + self.w2) if (self.w1 + self.w2) > 0 else 0.0
        )
        self.FACTOR = 15.46484755
        self.rho = self.FACTOR * np.sqrt(self.w / 2.0) * self.DQ

        self.Erel1 = 0.5 * (self.FACTOR**2) * (self.w1**2) * (self.DQ**2)
        self.Erel2 = 0.5 * (self.FACTOR**2) * (self.w2**2) * (self.DQ**2)
        self.sinfi = (
            np.sqrt(self.w2 / (self.w1 + self.w2)) if (self.w1 + self.w2) > 0 else 0.0
        )
        self.cosfi = (
            np.sqrt(self.w1 / (self.w1 + self.w2)) if (self.w1 + self.w2) > 0 else 0.0
        )

        print(f"Relaxation energy in ground state: {self.Erel1:.6f} eV")
        print(f"Relaxation energy in excited state: {self.Erel2:.6f} eV")

        self.overlap_matrix = np.zeros((self.NN1 + 1, self.NN2 + 1))

    def compute_overlap_matrix(self) -> None:
        """Populate transition overlap matrix mapping states configuration."""
        for i in range(self.NN1 + 1):
            for j in range(self.NN2 + 1):
                self.overlap_matrix[i, j] = utils.calculate_overlap_element(
                    i, j, self.rho, self.cosfi, self.sinfi
                )

    def compute_spectrum(self) -> None:
        """Compute individual transition intensities and check closure."""
        self.compute_overlap_matrix()

        # Guard against zero temperature limits
        if self.TE <= 0:
            Z = 1.0
            weights = np.zeros(self.NN1 + 1)
            weights[0] = 1.0
        else:
            Z = 1.0 / (1.0 - np.exp(-self.w1 / self.TE))
            weights = np.exp(-np.arange(self.NN1 + 1) * self.w1 / self.TE) / Z

        contr, en = [], []
        for i in range(self.NN1 + 1):
            weight = weights[i]
            for j in range(self.NN2 + 1):
                val = self.overlap_matrix[i, j]
                contrib = weight * (val**2)
                contr.append(contrib)
                en.append(self.EZPL - j * self.w2 + i * self.w1)

        print(f"Closure relation sum (should be ~1.0): {sum(contr):.6f}")
        self.overlap_data = {"contributions": contr, "energies": en}

    def compute_lineshape(self) -> None:
        """Compute Gaussian-broadened density of states (DOS) and luminescence curves."""
        if not self.overlap_data:
            self.compute_spectrum()

        energies_grid = self.E0 + np.arange(self.M) * self.dE
        contr = np.array(self.overlap_data["contributions"])
        en = np.array(self.overlap_data["energies"])

        # Vectorized calculation pipeline across grid positions to maximize performance
        delta_E = en[:, np.newaxis] - energies_grid[np.newaxis, :]
        gaussian = np.exp(-(delta_E**2) / (2 * self.sigma**2)) / (
            self.sigma * np.sqrt(2 * np.pi)
        )
        dos = np.dot(contr, gaussian)

        dosw3 = dos * (energies_grid**3)
        norm_factor = np.sum(dosw3) * self.dE
        if norm_factor > 0:
            dosw3 /= norm_factor

        self.spectral_data = {
            "energies": energies_grid.tolist(),
            "dos": dos.tolist(),
            "dosw3": dosw3.tolist(),
        }

    def save_results(
        self, overlap_file: str = "overlap.json", lineshape_file: str = "lineshape.json"
    ) -> None:
        """Serialize parameters and calculated datasets directly into target JSON files."""
        Path(overlap_file).write_text(self.to_json(), encoding="utf-8")
        with open(lineshape_file, "w", encoding="utf-8") as f:
            json.dump(self.spectral_data, f, indent=4)

    def plot_lineshape(
        self, save_file: Optional[str] = None, figsize: Tuple[float, float] = (4.4, 4.4)
    ) -> None:
        """Generate visual breakdown plots detailing the normalized intensity spectrum."""
        if not self.spectral_data:
            raise ValueError("Run compute_lineshape() before plotting.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.spectral_data["energies"], self.spectral_data["dosw3"])
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity (arb. u.)")
        ax.set_yticks([])

        if save_file:
            plt.savefig(save_file, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Lineshape plot saved to {save_file}")
        else:
            plt.show()
            plt.close(fig)

    def get_peak_position(self) -> Tuple[float, float]:
        """Fetch the energy location corresponding to the peak maximum intensity."""
        if not self.spectral_data:
            raise ValueError("Run compute_lineshape() before accessing metrics.")
        dosw3 = np.array(self.spectral_data["dosw3"])
        energies = np.array(self.spectral_data["energies"])
        idx_max = np.argmax(dosw3)
        print(f"Peak position: {energies[idx_max]:.3f} eV at {self.T} K.")
        return float(energies[idx_max]), float(dosw3[idx_max])

    def get_fwhm(self) -> float:
        """Compute the Full Width at Half Maximum (FWHM) metrics from the normalized array."""
        if not self.spectral_data:
            raise ValueError("Run compute_lineshape() before accessing metrics.")
        dosw3 = np.array(self.spectral_data["dosw3"])
        energies = np.array(self.spectral_data["energies"])

        half_max = np.max(dosw3) / 2.0
        indices = np.where(dosw3 >= half_max)[0]

        if len(indices) < 2:
            return 0.0

        fwhm = energies[indices[-1]] - energies[indices[0]]
        print(f"FWHM: {fwhm:.3f} eV at {self.T} K.")
        return float(fwhm)


@dataclass
class ConfigurationCoordinateDiagram(MSONable):
    """
    Class managing Potential Energy Surface calculations and diagrams.
    """

    ground_struct: Structure
    excited_struct: Structure
    dQ: float = field(init=False)

    def __post_init__(self):
        """Calculate secondary equilibrium distance vectors automatically."""
        self.dQ = calc_delta_Q(self.ground_struct, self.excited_struct)

    def generate_structures(
        self, displacements: Union[List[float], np.ndarray], remove_zero: bool = True
    ) -> Tuple[List[Structure], List[Structure]]:
        """Linearly interpolate atomic displacement paths across configurations."""
        disp_arr = np.atleast_1d(displacements)
        if remove_zero:
            disp_arr = disp_arr[disp_arr != 0.0]

        ground_structs = self.ground_struct.interpolate(
            self.excited_struct, nimages=disp_arr
        )
        excited_structs = self.ground_struct.interpolate(
            self.excited_struct, nimages=(disp_arr + 1.0)
        )
        return ground_structs, excited_structs

    def setup_calculations(
        self,
        displacements: Union[List[float], np.ndarray],
        output_dir: Union[str, Path],
        ground_input_dir: Union[str, Path],
        excited_input_dir: Union[str, Path],
        input_files: Optional[List[str]] = None,
    ) -> None:
        """Generate file trees containing interpolated VASP calculation parameters."""
        if input_files is None:
            input_files = ["KPOINTS", "POTCAR", "INCAR"]

        out_path = Path(output_dir)
        g_in_path = Path(ground_input_dir)
        e_in_path = Path(excited_input_dir)

        g_structs, e_structs = self.generate_structures(displacements)

        for idx, struct in enumerate(g_structs):
            target_dir = out_path / "ground" / str(idx)
            target_dir.mkdir(parents=True, exist_ok=True)
            struct.to(filename=str(target_dir / "POSCAR"), fmt="poscar")
            for filename in input_files:
                copyfile(g_in_path / filename, target_dir / filename)

        for idx, struct in enumerate(e_structs):
            target_dir = out_path / "excited" / str(idx)
            target_dir.mkdir(parents=True, exist_ok=True)
            struct.to(filename=str(target_dir / "POSCAR"), fmt="poscar")
            for filename in input_files:
                copyfile(e_in_path / filename, target_dir / filename)

    def extract_pes_profile(
        self, vasprun_paths: List[Union[str, Path]], tol: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse raw output collection files to construct the local energy surface."""
        from pymatgen.io.vasp.outputs import Vasprun

        total_runs = len(vasprun_paths)
        q_values = np.zeros(total_runs)
        energies = np.zeros(total_runs)

        for idx, path in enumerate(vasprun_paths):
            vr = Vasprun(str(path), parse_dos=False, parse_eigen=False)
            q_values[idx] = get_q_from_structure(
                self.ground_struct, self.excited_struct, vr.structures[-1], tol=tol
            )
            energies[idx] = vr.final_energy

        return q_values, (energies - np.min(energies))

    def analyze_ccd(
        self,
        ground_vaspruns: List[Union[str, Path]],
        excited_vaspruns: List[Union[str, Path]],
        dE: float = 0.0,
        plot: bool = True,
        figsize: Tuple[float, float] = (3.3, 3.3),
        xlim: Tuple[float, float] = (-3.0, 10.0),
        ylim: Tuple[float, float] = (-0.5, 4.0),
        save_plot: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Process completed calculation data arrays and fit harmonic wells."""
        q_ground, e_ground = self.extract_pes_profile(ground_vaspruns)
        q_excited, e_excited = self.extract_pes_profile(excited_vaspruns)
        e_excited += dE

        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(q_ground, e_ground, label="Ground State")
            ax.scatter(q_excited, e_excited, label="Excited State")

            grid_line = np.linspace(xlim[0], xlim[1], 100)
            ground_omega = utils.get_omega_from_pes(
                q_ground, e_ground, ax=ax, eval_grid=grid_line
            )
            excited_omega = utils.get_omega_from_pes(
                q_excited, e_excited, ax=ax, eval_grid=grid_line
            )

            ax.set_xlabel(r"$Q$ ($\mathrm{amu}^{1/2}\cdot\mathrm{\AA}$)")
            ax.set_ylabel("Energy (eV)")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.legend()

            if save_plot:
                plt.savefig(save_plot, bbox_inches="tight", dpi=600)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)
        else:
            ground_omega = utils.get_omega_from_pes(q_ground, e_ground)
            excited_omega = utils.get_omega_from_pes(q_excited, e_excited)

        return ground_omega, excited_omega

    def estimate_vertical_transitions(
        self, ground_omega: float, excited_omega: float, dE: float, eps: float = 1e-6
    ) -> Tuple[float, float, float, float]:
        """Calculate vertical absorption and emission energy transitions."""
        if ground_omega < eps or excited_omega < eps:
            raise ValueError("Phonon frequencies must be strictly positive.")

        conversion_factor = (self.dQ**2) * AMU2KG * (ANG2M**2) / (HBAR_EVS**2) / EV2J

        fc_e = 0.5 * (excited_omega**2) * conversion_factor
        fc_g = 0.5 * (ground_omega**2) * conversion_factor

        e_abs = fc_e + dE
        e_em = dE - fc_g

        print(f"Absorption Energy: {e_abs:.6f} eV")
        print(f"Emission Energy:   {e_em:.6f} eV")
        print(f"Franck-Condon shift (excited): {fc_e:.6f} eV")
        print(f"Franck-Condon shift (ground):  {fc_g:.6f} eV")

        return e_abs, e_em, fc_e, fc_g
