# -*- coding: utf-8 -*-
"""
Defect Optical Properties Engine (DefectPL) core module.
Authors: Shibu Meher, Manoj Dey
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from monty.json import MSONable

if TYPE_CHECKING:
    from pymatgen.core import Structure

from defectpl.constants import AMU2KG, ANG2M, EV2J, HBAR_EVS
from defectpl.plot import Plotter
import defectpl.utils as utils
from defectpl.io.vasp import calc_delta_Q, get_q_from_structure


@dataclass
class Photoluminescence(MSONable):
    """
    Core engine for first-principles photoluminescence lineshape calculations.

    Implements the generating-function formalism of Alkauskas *et al.*
    (*New J. Phys.* **16**, 073026, 2014) to compute the full multi-phonon
    PL spectrum of a point defect from DFT inputs.

    Two input modes are supported:

    * **Displacement mode** — supply ``dR`` (atomic displacements between
      ground and excited equilibrium geometries).
    * **Force mode** — supply ``dF`` (forces on the ground-state atoms when
      the system is in the excited-state charge state).

    Parameters
    ----------
    frequencies : numpy.ndarray, shape (nmodes,)
        Phonon mode frequencies at the Γ point in **eV**.
    eigenvectors : numpy.ndarray, shape (nmodes, natoms, 3)
        Mass-normalised phonon displacement eigenvectors (real part only).
    masses : numpy.ndarray, shape (natoms,)
        Atomic masses in atomic mass units (amu).
    EZPL : float
        Zero-phonon line (ZPL) energy in **eV**.
    dR : numpy.ndarray, shape (natoms, 3), optional
        Atomic displacement vector (excited − ground equilibrium) in **Å**.
        Used in displacement mode.
    dF : numpy.ndarray, shape (natoms, 3), optional
        Atomic force difference (excited charge − ground charge) in **eV/Å**.
        Used in force mode.  ``dR`` takes priority if both are given.
    resolution : int, optional
        Number of energy grid points per eV; sets spectral resolution.
        Default 1000.
    max_energy : float, optional
        Upper bound of the energy axis in **eV**.  Default 5.0.
    sigma : float or (float, float), optional
        Gaussian broadening width applied to the spectral function S(ω) in **eV**.
        Scalar → uniform broadening for all modes.
        2-tuple (sigma_low, sigma_high) → linearly interpolated from the lowest
        to the highest phonon frequency (Jin *et al.* 2021, Fig. 5).
        Default 6 meV.
    gamma : float, optional
        Lorentzian (homogeneous) broadening of the ZPL in **meV**.
        Default 2.0 meV.
    temperature : float, optional
        Lattice temperature in **Kelvin** used for the Bose-Einstein phonon
        occupation :math:`\\bar{n}_k(T)`.  Pass 0 (default) to reproduce the
        T = 0 K limit where the thermal correction vanishes and the result
        is identical to the original Alkauskas (2014) formula.

    Attributes
    ----------
    natoms : int
        Number of atoms in the supercell.
    delR : float
        Root-mean-square atomic displacement ΔR in **Å**.
    delQ : float
        Mass-weighted configuration coordinate displacement ΔQ in
        amu\\ :sup:`1/2`·Å.
    qks : numpy.ndarray, shape (nmodes,)
        Mode-projected configurational displacement q\\ :sub:`k` in
        amu\\ :sup:`1/2`·Å.
    Sks : numpy.ndarray, shape (nmodes,)
        Partial (mode-resolved) Huang–Rhys factors S\\ :sub:`k`.
    HR_factor : float
        Total Huang–Rhys factor S = Σ S\\ :sub:`k`.
    DW_factor : float
        Debye–Waller factor exp(−S).
    iprs : numpy.ndarray, shape (nmodes,)
        Phonon inverse participation ratio (traditional convention): Σp²/(Σp)²;
        range [1/N, 1] where large = localized.
    iprs_alkauskas : numpy.ndarray, shape (nmodes,)
        Phonon IPR following Alkauskas *et al.* (2014) eq. 12: (Σp)²/Σp²;
        range [1, N] where small = localized.  Reciprocal of ``iprs``.
    localization_ratio : numpy.ndarray, shape (nmodes,)
        Localization ratio β\\ :sub:`k` = N · IPR\\ :sub:`k`; range [1, N].
    nks : numpy.ndarray, shape (nmodes,)
        Bose-Einstein phonon occupation numbers :math:`\\bar{n}_k(T)`.
        All zeros at T = 0.
    C_omega : numpy.ndarray
        Thermal electron–phonon spectral density :math:`C(\\hbar\\omega, T)` in
        eV\\ :sup:`-1`.  Zero array at T = 0.
    Cts : numpy.ndarray
        Real-valued time-domain thermal correction :math:`C(t, T)`.
    C_total : float
        Zero-time thermal correction :math:`C(0,T) = \\sum_k \\bar{n}_k S_k`.
        Zero at T = 0.
    effective_phonon_freq : float
        Effective phonon frequency Ω in **eV** (Jin *et al.* 2021, Eq. 16),
        computed as the displacement-weighted average :math:`\\Omega = \\sum_k \\omega_k^2
        \\Delta Q_k^2 / \\sum_k \\Delta Q_k^2`.
    S_omega : numpy.ndarray
        Continuous Huang–Rhys spectral density S(ω) (eV\\ :sup:`-1`).
    Sts : numpy.ndarray
        Fourier transform of S(ω) used in the generating function.
    Gts : numpy.ndarray
        Generating function G(t) = exp[S(t) − S] · exp(−γ|t|).
    A_line : numpy.ndarray
        Photon energy axis for the lineshape in **eV**.
    intensity : numpy.ndarray
        Normalised PL intensity as a function of photon energy.

    Notes
    -----
    The PL lineshape is obtained via the Fourier transform of the
    generating function G(t):

    .. math::

        L(\\hbar\\omega) = \\int_{-\\infty}^{\\infty} G(t)\\,
        e^{i(E_{\\mathrm{ZPL}} - \\hbar\\omega)t/\\hbar}\\, dt

    where

    .. math::

        G(t) = e^{S(t) - S} \\cdot e^{-\\gamma|t|}

    and :math:`S(t) = \\int_0^{\\infty} S(\\hbar\\omega)\\,
    e^{-i\\omega t}\\, d(\\hbar\\omega)`.

    References
    ----------
    Alkauskas, Buckley, Awschalom & Van de Walle,
    *New J. Phys.* **16**, 073026 (2014).

    Examples
    --------
    >>> from defectpl.phonon import read_band_yaml
    >>> import numpy as np
    >>> freqs, evecs, masses = read_band_yaml("band.yaml")
    >>> dR = np.load("dR.npy")          # shape (natoms, 3), in Å
    >>> pl = Photoluminescence(
    ...     frequencies=freqs,
    ...     eigenvectors=evecs,
    ...     masses=masses,
    ...     EZPL=1.945,
    ...     dR=dR,
    ... )
    >>> print(f"S = {pl.HR_factor:.3f},  DW = {pl.DW_factor:.4f}")
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
    sigma: Union[float, Tuple[float, float]] = (
        6e-3  # Broadening: scalar or (low, high) eV
    )
    gamma: float = 2.0  # Homogeneous/inhomogeneous ZPL broadening factor
    temperature: float = 0.0  # Lattice temperature in K (0 = T=0 limit)

    # Dependent calculated properties stored dynamically downstream
    natoms: int = field(init=False)
    delR: float = field(init=False, default=None)
    delQ: float = field(init=False, default=None)
    qks: np.ndarray = field(init=False, default=None)
    Sks: np.ndarray = field(init=False, default=None)
    HR_factor: float = field(init=False, default=None)
    DW_factor: float = field(init=False, default=None)
    iprs: np.ndarray = field(init=False, default=None)
    iprs_alkauskas: np.ndarray = field(init=False, default=None)
    localization_ratio: np.ndarray = field(init=False, default=None)
    nks: np.ndarray = field(init=False, default=None)
    C_omega: np.ndarray = field(init=False, default=None)
    Cts: np.ndarray = field(init=False, default=None)
    C_total: float = field(init=False, default=0.0)
    effective_phonon_freq: float = field(init=False, default=None)
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
        """Run the full calculation pipeline and populate all derived attributes."""
        self.delR = utils.calc_delR(self.dR) if self.dR is not None else 0.0
        self.delQ = (
            utils.calc_delQ(self.masses, self.dR) if self.dR is not None else 0.0
        )

        if self.dF is not None and np.any(self.dF):
            self.qks = utils.calc_qks_force_vectorized(
                self.masses, self.dF, self.eigenvectors, self.frequencies
            )
        elif self.dR is not None and np.any(self.dR):
            self.qks = utils.calc_qks_vectorized(
                self.masses, self.dR, self.eigenvectors
            )
        else:
            raise ValueError(
                "Either dR or dF must be provided and non-zero to compute qks."
            )

        self.Sks = utils.calc_Sks(self.qks, self.frequencies)
        self.HR_factor = float(np.sum(self.Sks))
        self.DW_factor = float(np.exp(-self.HR_factor))

        self.iprs = utils.calc_IPR(self.eigenvectors)
        self.iprs_alkauskas = utils.calc_IPR_alkauskas(self.eigenvectors)
        self.localization_ratio = self.natoms * self.iprs

        self.nks = utils.calc_phonon_occupation(self.frequencies, self.temperature)
        self.C_omega = utils.calc_C_omega(
            self.frequencies, self.Sks, self.nks, self.omega_range, self.sigma
        )
        self.Cts = utils.calc_Ct(self.C_omega)
        self.C_total = utils.calc_C_total(self.nks, self.Sks)
        self.effective_phonon_freq = utils.calc_effective_phonon_frequency(
            self.frequencies, self.qks
        )

        self.S_omega = utils.calc_S_omega(
            self.frequencies, self.Sks, self.omega_range, self.sigma
        )
        self.Sts = utils.calc_St(self.S_omega)
        self.Gts = utils.calc_Gts(
            self.Sts,
            self.HR_factor,
            self.gamma,
            self.resolution,
            Cts=self.Cts,
            C_total=self.C_total,
        )
        self.A_line, self.intensity = utils.calc_Spectrum_Intensity(
            self.Gts, self.EZPL, self.resolution
        )

    def as_dict(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary.

        Complex-valued arrays (``Sts``, ``Gts``) and derived spectral arrays
        (``A_line``, ``intensity``) are intentionally omitted; they are
        cheaply recomputed by :meth:`from_dict`.
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
            "sigma": list(self.sigma) if hasattr(self.sigma, "__len__") else self.sigma,
            "temperature": self.temperature,
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
            "iprs_alkauskas": (
                self.iprs_alkauskas.tolist()
                if self.iprs_alkauskas is not None
                else None
            ),
            "localization_ratio": (
                self.localization_ratio.tolist()
                if self.localization_ratio is not None
                else None
            ),
            "nks": self.nks.tolist() if self.nks is not None else None,
            "C_omega": self.C_omega.tolist() if self.C_omega is not None else None,
            "C_total": float(self.C_total),
            "effective_phonon_freq": (
                float(self.effective_phonon_freq)
                if self.effective_phonon_freq is not None
                else None
            ),
            "omega_range": self.omega_range,
            "S_omega": self.S_omega.tolist() if self.S_omega is not None else None,
            # Drop complex/spectral arrays — cheaply recomputed by from_dict
            "Sts": None,
            "Gts": None,
            "A_line": None,
            "intensity": None,
            "Cts": None,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """
        Deserialize from a dictionary produced by :meth:`as_dict`.

        Core inputs and real-valued computed properties are loaded directly;
        complex arrays (``Sts``, ``Gts``) and the intensity spectrum are
        recomputed on the fly from the stored S(ω).
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
        _sigma = d.get("sigma", 6e-3)
        obj.sigma = tuple(_sigma) if isinstance(_sigma, list) else _sigma
        obj.temperature = d.get("temperature", 0.0)

        # Load Stored Real-Valued Properties
        obj.natoms = d.get("natoms", len(obj.masses))
        obj.delR = d.get("delR")
        obj.delQ = d.get("delQ")
        obj.qks = np.array(d["qks"]) if d.get("qks") is not None else None
        obj.Sks = np.array(d["Sks"]) if d.get("Sks") is not None else None
        obj.HR_factor = d.get("HR_factor")
        obj.DW_factor = d.get("DW_factor")
        obj.iprs = np.array(d["iprs"]) if d.get("iprs") is not None else None
        obj.iprs_alkauskas = (
            np.array(d["iprs_alkauskas"])
            if d.get("iprs_alkauskas") is not None
            else None
        )
        obj.localization_ratio = (
            np.array(d["localization_ratio"])
            if d.get("localization_ratio") is not None
            else None
        )
        obj.nks = np.array(d["nks"]) if d.get("nks") is not None else None
        obj.C_omega = np.array(d["C_omega"]) if d.get("C_omega") is not None else None
        obj.C_total = float(d.get("C_total", 0.0))
        obj.effective_phonon_freq = d.get("effective_phonon_freq")
        obj.omega_range = d.get(
            "omega_range", [0.0, obj.max_energy, int(obj.max_energy * obj.resolution)]
        )
        obj.S_omega = np.array(d["S_omega"]) if d.get("S_omega") is not None else None

        # Placeholders for complex/derived arrays (recomputed below)
        obj.Sts = None
        obj.Gts = None
        obj.A_line = None
        obj.intensity = None
        obj.Cts = None

        # Recompute complex-dependent pipeline from stored S_omega and C_omega
        if obj.intensity is None:
            obj.Cts = utils.calc_Ct(obj.C_omega) if obj.C_omega is not None else None
            obj.Sts = utils.calc_St(obj.S_omega)
            obj.Gts = utils.calc_Gts(
                obj.Sts,
                obj.HR_factor,
                obj.gamma,
                obj.resolution,
                Cts=obj.Cts,
                C_total=obj.C_total,
            )
            obj.A_line, obj.intensity = utils.calc_Spectrum_Intensity(
                obj.Gts, obj.EZPL, obj.resolution
            )

        return obj

    @classmethod
    def from_dict_expensive(cls, d: dict):
        """
        Reconstruct by replaying the full pipeline from primary inputs only.

        Slower than :meth:`from_dict` because ``__post_init__`` recomputes
        everything from scratch; useful when stored arrays may be stale.
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
            temperature=d.get("temperature", 0.0),
        )

    def generate_plots(
        self,
        out_dir: Union[str, Path],
        max_freq: Optional[float] = None,
        iylim=None,
        fig_format="pdf",
    ):
        r"""
        Generate all standard diagnostic plots and save them to *out_dir*.

        Produces fourteen figures: phonon energy vs mode index, traditional IPR vs energy,
        Alkauskas IPR vs energy, localization ratio vs energy, phonon occupation vs energy,
        q\ :sub:`k` vs energy, S\ :sub:`k` vs energy, S(ω) alone, C(ω,T) alone,
        S(ω)+S\ :sub:`k`, S(ω)+S\ :sub:`k` coloured by localization ratio,
        S(ω)+S\ :sub:`k` coloured by traditional IPR, S(ω)+S\ :sub:`k` coloured by
        Alkauskas IPR, and the final PL intensity spectrum.  Absorption plots require
        a :class:`Photoabsorption` object (which uses excited-state phonons).

        Parameters
        ----------
        out_dir : str or Path
            Directory where figure files are written (created if absent).
        max_freq : float, optional
            Upper frequency cut-off for S(ω) plots in **meV**.  If ``None``
            the maximum phonon frequency is used.
        iylim : tuple of float, optional
            ``(ymin, ymax)`` for the intensity plot y-axis.
        fig_format : {'pdf', 'png', 'svg', 'jpg'}, optional
            Output figure format.  Default ``'pdf'``.
        """
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
        plotter.plot_ipr_alkauskas_vs_penergy(
            self.frequencies,
            self.iprs_alkauskas,
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
        plotter.plot_S_omega_Sks_ipr_alkauskas_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.iprs_alkauskas,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_nk_vs_penergy(
            self.frequencies,
            self.nks,
            self.temperature,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_C_omega_vs_penergy(
            self.frequencies,
            self.C_omega,
            self.omega_range,
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
class Photoabsorption(MSONable):
    """
    Core engine for first-principles photoabsorption lineshape calculations.

    Implements the generating-function formalism of Alkauskas *et al.*
    (*New J. Phys.* **16**, 073026, 2014) to compute the full multi-phonon
    photoabsorption spectrum of a point defect from DFT inputs.

    This class uses **excited-state phonons** (a phonopy ``band.yaml`` run on
    the excited-state geometry) to correctly describe the coupling of the
    photon-absorption process to vibrational modes.  This is physically
    distinct from :class:`Photoluminescence`, which uses ground-state phonons
    for PL emission.

    Two input modes are supported:

    * **Displacement mode** — supply ``dR`` (atomic displacements between
      ground and excited equilibrium geometries).
    * **Force mode** — supply ``dF`` (forces on the excited-state atoms when
      the system is in the ground-state charge state).

    Parameters
    ----------
    frequencies : numpy.ndarray, shape (nmodes,)
        Phonon mode frequencies at the Γ point in **eV** (from excited-state
        phonopy calculation).
    eigenvectors : numpy.ndarray, shape (nmodes, natoms, 3)
        Mass-normalised phonon displacement eigenvectors (real part only).
    masses : numpy.ndarray, shape (natoms,)
        Atomic masses in atomic mass units (amu).
    EZPL : float
        Zero-phonon line (ZPL) energy in **eV**.
    dR : numpy.ndarray, shape (natoms, 3), optional
        Atomic displacement vector (excited − ground equilibrium) in **Å**.
        Used in displacement mode.
    dF : numpy.ndarray, shape (natoms, 3), optional
        Atomic force difference (ground charge − excited charge) evaluated at
        the ES geometry in **eV/Å**.  Used in force mode.  ``dR`` takes
        priority if both are given.
    resolution : int, optional
        Number of energy grid points per eV; sets spectral resolution.
        Default 1000.
    max_energy : float, optional
        Upper bound of the energy axis in **eV**.  Default 5.0.
    sigma : float or (float, float), optional
        Gaussian broadening width applied to the spectral function S(ω) in **eV**.
        Default 6 meV.
    gamma : float, optional
        Lorentzian (homogeneous) broadening of the ZPL in **meV**.
        Default 2.0 meV.
    temperature : float, optional
        Lattice temperature in **Kelvin**.  Pass 0 (default) for the T = 0 K limit.

    Attributes
    ----------
    natoms : int
        Number of atoms in the supercell.
    delR : float
        Root-mean-square atomic displacement ΔR in **Å**.
    delQ : float
        Mass-weighted configuration coordinate displacement ΔQ in
        amu\\ :sup:`1/2`·Å.
    qks : numpy.ndarray, shape (nmodes,)
        Mode-projected configurational displacement q\\ :sub:`k`.
    Sks : numpy.ndarray, shape (nmodes,)
        Partial (mode-resolved) Huang–Rhys factors S\\ :sub:`k`.
    HR_factor : float
        Total Huang–Rhys factor S = Σ S\\ :sub:`k`.
    DW_factor : float
        Debye–Waller factor exp(−S).
    iprs : numpy.ndarray, shape (nmodes,)
        Phonon inverse participation ratio (traditional convention).
    iprs_alkauskas : numpy.ndarray, shape (nmodes,)
        Phonon IPR following Alkauskas *et al.* (2014) eq. 12.
    localization_ratio : numpy.ndarray, shape (nmodes,)
        Localization ratio β\\ :sub:`k` = N · IPR\\ :sub:`k`.
    nks : numpy.ndarray, shape (nmodes,)
        Bose-Einstein phonon occupation numbers :math:`\\bar{n}_k(T)`.
    C_omega : numpy.ndarray
        Thermal electron–phonon spectral density :math:`C(\\hbar\\omega, T)`.
    Cts : numpy.ndarray
        Real-valued time-domain thermal correction :math:`C(t, T)`.
    C_total : float
        Zero-time thermal correction :math:`C(0,T) = \\sum_k \\bar{n}_k S_k`.
    effective_phonon_freq : float
        Effective phonon frequency Ω in **eV**.
    S_omega : numpy.ndarray
        Continuous Huang–Rhys spectral density S(ω) (eV\\ :sup:`-1`).
    Sts : numpy.ndarray
        Fourier transform of S(ω) used in the generating function.
    Gts : numpy.ndarray
        Generating function G(t) = exp[S(t) − S] · exp(−γ|t|).
    A_abs : numpy.ndarray
        Absorption spectral function.
    absorption : numpy.ndarray
        Normalised absorption intensity :math:`\\alpha(\\hbar\\omega) \\propto \\omega\\, A_{abs}`.

    Notes
    -----
    **Physics rationale**: In the Franck–Condon picture, the absorption
    lineshape couples to the vibrational modes of the *excited* electronic
    state, because the nuclear wavepacket created by photon absorption
    evolves on the excited-state potential energy surface.  Conversely, PL
    emission couples to *ground-state* phonons.  Using the correct phonon set
    (ES for absorption, GS for emission) is essential for quantitatively
    accurate lineshapes, particularly when the two geometries differ
    significantly.

    References
    ----------
    Alkauskas, Buckley, Awschalom & Van de Walle,
    *New J. Phys.* **16**, 073026 (2014).

    Examples
    --------
    >>> from defectpl.phonon import read_band_yaml
    >>> import numpy as np
    >>> freqs_es, evecs_es, masses = read_band_yaml("band_es.yaml")
    >>> dR = np.load("dR.npy")
    >>> abs_engine = Photoabsorption(
    ...     frequencies=freqs_es,
    ...     eigenvectors=evecs_es,
    ...     masses=masses,
    ...     EZPL=1.945,
    ...     dR=dR,
    ... )
    >>> print(f"S = {abs_engine.HR_factor:.3f}")
    """

    # 1. Mandatory Core Inputs
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    masses: np.ndarray
    EZPL: float

    # 2. Runtime Optional Parameters
    dR: Optional[np.ndarray] = None
    dF: Optional[np.ndarray] = None
    resolution: int = 1000
    max_energy: float = 5.0
    sigma: Union[float, Tuple[float, float]] = 6e-3
    gamma: float = 2.0
    temperature: float = 0.0

    # Dependent calculated properties
    natoms: int = field(init=False)
    delR: float = field(init=False, default=None)
    delQ: float = field(init=False, default=None)
    qks: np.ndarray = field(init=False, default=None)
    Sks: np.ndarray = field(init=False, default=None)
    HR_factor: float = field(init=False, default=None)
    DW_factor: float = field(init=False, default=None)
    iprs: np.ndarray = field(init=False, default=None)
    iprs_alkauskas: np.ndarray = field(init=False, default=None)
    localization_ratio: np.ndarray = field(init=False, default=None)
    nks: np.ndarray = field(init=False, default=None)
    C_omega: np.ndarray = field(init=False, default=None)
    Cts: np.ndarray = field(init=False, default=None)
    C_total: float = field(init=False, default=0.0)
    effective_phonon_freq: float = field(init=False, default=None)
    omega_range: List[Union[float, int]] = field(init=False, default=None)
    S_omega: np.ndarray = field(init=False, default=None)
    Sts: np.ndarray = field(init=False, default=None)
    Gts: np.ndarray = field(init=False, default=None)
    A_abs: np.ndarray = field(init=False, default=None)
    absorption: np.ndarray = field(init=False, default=None)

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
        """Run the full calculation pipeline and populate all derived attributes."""
        self.delR = utils.calc_delR(self.dR) if self.dR is not None else 0.0
        self.delQ = (
            utils.calc_delQ(self.masses, self.dR) if self.dR is not None else 0.0
        )

        if self.dF is not None and np.any(self.dF):
            self.qks = utils.calc_qks_force_vectorized(
                self.masses, self.dF, self.eigenvectors, self.frequencies
            )
        elif self.dR is not None and np.any(self.dR):
            self.qks = utils.calc_qks_vectorized(
                self.masses, self.dR, self.eigenvectors
            )
        else:
            raise ValueError(
                "Either dR or dF must be provided and non-zero to compute qks."
            )

        self.Sks = utils.calc_Sks(self.qks, self.frequencies)
        self.HR_factor = float(np.sum(self.Sks))
        self.DW_factor = float(np.exp(-self.HR_factor))

        self.iprs = utils.calc_IPR(self.eigenvectors)
        self.iprs_alkauskas = utils.calc_IPR_alkauskas(self.eigenvectors)
        self.localization_ratio = self.natoms * self.iprs

        self.nks = utils.calc_phonon_occupation(self.frequencies, self.temperature)
        self.C_omega = utils.calc_C_omega(
            self.frequencies, self.Sks, self.nks, self.omega_range, self.sigma
        )
        self.Cts = utils.calc_Ct(self.C_omega)
        self.C_total = utils.calc_C_total(self.nks, self.Sks)
        self.effective_phonon_freq = utils.calc_effective_phonon_frequency(
            self.frequencies, self.qks
        )

        self.S_omega = utils.calc_S_omega(
            self.frequencies, self.Sks, self.omega_range, self.sigma
        )
        self.Sts = utils.calc_St(self.S_omega)
        self.Gts = utils.calc_Gts(
            self.Sts,
            self.HR_factor,
            self.gamma,
            self.resolution,
            Cts=self.Cts,
            C_total=self.C_total,
        )
        self.A_abs, self.absorption = utils.calc_Absorption_Intensity(
            self.Gts, self.EZPL, self.resolution
        )

    def as_dict(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary.

        Complex-valued arrays (``Sts``, ``Gts``) and derived spectral arrays
        (``A_abs``, ``absorption``) are stored; Sts/Gts are cheaply recomputed
        by :meth:`from_dict`.
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
            "sigma": list(self.sigma) if hasattr(self.sigma, "__len__") else self.sigma,
            "temperature": self.temperature,
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
            "iprs_alkauskas": (
                self.iprs_alkauskas.tolist()
                if self.iprs_alkauskas is not None
                else None
            ),
            "localization_ratio": (
                self.localization_ratio.tolist()
                if self.localization_ratio is not None
                else None
            ),
            "nks": self.nks.tolist() if self.nks is not None else None,
            "C_omega": self.C_omega.tolist() if self.C_omega is not None else None,
            "C_total": float(self.C_total),
            "effective_phonon_freq": (
                float(self.effective_phonon_freq)
                if self.effective_phonon_freq is not None
                else None
            ),
            "omega_range": self.omega_range,
            "S_omega": self.S_omega.tolist() if self.S_omega is not None else None,
            "absorption": (
                self.absorption.tolist() if self.absorption is not None else None
            ),
            # Drop complex/spectral arrays — cheaply recomputed by from_dict
            "Sts": None,
            "Gts": None,
            "A_abs": None,
            "Cts": None,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """
        Deserialize from a dictionary produced by :meth:`as_dict`.

        Core inputs and real-valued computed properties are loaded directly;
        complex arrays (``Sts``, ``Gts``) and the absorption spectrum are
        recomputed on the fly from the stored S(ω).
        """
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
        _sigma = d.get("sigma", 6e-3)
        obj.sigma = tuple(_sigma) if isinstance(_sigma, list) else _sigma
        obj.temperature = d.get("temperature", 0.0)

        # Load Stored Real-Valued Properties
        obj.natoms = d.get("natoms", len(obj.masses))
        obj.delR = d.get("delR")
        obj.delQ = d.get("delQ")
        obj.qks = np.array(d["qks"]) if d.get("qks") is not None else None
        obj.Sks = np.array(d["Sks"]) if d.get("Sks") is not None else None
        obj.HR_factor = d.get("HR_factor")
        obj.DW_factor = d.get("DW_factor")
        obj.iprs = np.array(d["iprs"]) if d.get("iprs") is not None else None
        obj.iprs_alkauskas = (
            np.array(d["iprs_alkauskas"])
            if d.get("iprs_alkauskas") is not None
            else None
        )
        obj.localization_ratio = (
            np.array(d["localization_ratio"])
            if d.get("localization_ratio") is not None
            else None
        )
        obj.nks = np.array(d["nks"]) if d.get("nks") is not None else None
        obj.C_omega = np.array(d["C_omega"]) if d.get("C_omega") is not None else None
        obj.C_total = float(d.get("C_total", 0.0))
        obj.effective_phonon_freq = d.get("effective_phonon_freq")
        obj.omega_range = d.get(
            "omega_range", [0.0, obj.max_energy, int(obj.max_energy * obj.resolution)]
        )
        obj.S_omega = np.array(d["S_omega"]) if d.get("S_omega") is not None else None
        obj.absorption = (
            np.array(d["absorption"]) if d.get("absorption") is not None else None
        )

        # Placeholders for complex/derived arrays (recomputed below)
        obj.Sts = None
        obj.Gts = None
        obj.A_abs = None
        obj.Cts = None

        # Recompute complex-dependent pipeline from stored S_omega and C_omega
        if obj.absorption is None:
            obj.Cts = utils.calc_Ct(obj.C_omega) if obj.C_omega is not None else None
            obj.Sts = utils.calc_St(obj.S_omega)
            obj.Gts = utils.calc_Gts(
                obj.Sts,
                obj.HR_factor,
                obj.gamma,
                obj.resolution,
                Cts=obj.Cts,
                C_total=obj.C_total,
            )
            obj.A_abs, obj.absorption = utils.calc_Absorption_Intensity(
                obj.Gts, obj.EZPL, obj.resolution
            )

        return obj

    @classmethod
    def from_dict_expensive(cls, d: dict):
        """
        Reconstruct by replaying the full pipeline from primary inputs only.

        Slower than :meth:`from_dict` because ``__post_init__`` recomputes
        everything from scratch; useful when stored arrays may be stale.
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
            temperature=d.get("temperature", 0.0),
        )

    def generate_plots(
        self,
        out_dir: Union[str, Path],
        max_freq: Optional[float] = None,
        iylim=None,
        fig_format="pdf",
    ):
        r"""
        Generate all standard diagnostic plots and save them to *out_dir*.

        Produces fourteen figures: phonon energy vs mode index, traditional IPR vs energy,
        Alkauskas IPR vs energy, localization ratio vs energy, phonon occupation vs energy,
        q\ :sub:`k` vs energy, S\ :sub:`k` vs energy, S(ω) alone, C(ω,T) alone,
        S(ω)+S\ :sub:`k`, S(ω)+S\ :sub:`k` coloured by localization ratio,
        S(ω)+S\ :sub:`k` coloured by traditional IPR, S(ω)+S\ :sub:`k` coloured by
        Alkauskas IPR, and the absorption spectrum.

        Parameters
        ----------
        out_dir : str or Path
            Directory where figure files are written (created if absent).
        max_freq : float, optional
            Upper frequency cut-off for S(ω) plots in **meV**.
        iylim : tuple of float, optional
            ``(ymin, ymax)`` for the absorption plot y-axis (unused; kept for
            API symmetry with :class:`Photoluminescence`).
        fig_format : {'pdf', 'png', 'svg', 'jpg'}, optional
            Output figure format.  Default ``'pdf'``.
        """
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
        plotter.plot_ipr_alkauskas_vs_penergy(
            self.frequencies,
            self.iprs_alkauskas,
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
        plotter.plot_S_omega_Sks_ipr_alkauskas_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.iprs_alkauskas,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_nk_vs_penergy(
            self.frequencies,
            self.nks,
            self.temperature,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        plotter.plot_C_omega_vs_penergy(
            self.frequencies,
            self.C_omega,
            self.omega_range,
            plot=False,
            out_dir=out_dir,
            max_freq=freq_limit,
            fig_format=fig_format,
        )
        plotter.plot_absorption_vs_penergy(
            frequencies=self.frequencies,
            absorption=self.absorption,
            resolution=self.resolution,
            xlim=iplot_xlim,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        print("All static visualization plots generated successfully.")


@dataclass
class VibrationalSpectra1D(MSONable):
    r"""
    1D harmonic-oscillator model for the vibrational lineshape of a luminescence band.

    Computes Franck–Condon overlaps between vibrational levels of two
    displaced harmonic potentials (ground and excited state) with
    potentially different effective frequencies, following
    Alkauskas, Yan & Van de Walle (*Phys. Rev. B* **90**, 075202, 2014).

    This model is complementary to :class:`Photoluminescence` — it replaces
    the multi-phonon generating function with an explicit sum over
    vibrational quantum numbers, which is exact within the harmonic
    approximation and more suitable when ω\ :sub:`1` ≠ ω\ :sub:`2`.

    Parameters
    ----------
    EZPL : float
        Zero-phonon line energy in **eV**.
    w1_meV : float
        Ground-state effective phonon frequency in **meV**.
    w2_meV : float
        Excited-state effective phonon frequency in **meV**.
    DQ : float
        Configuration coordinate displacement ΔQ in amu\ :sup:`1/2`·Å.
    T : float
        Temperature in **K** (sets Boltzmann weights for ground-state levels).
    E0 : float
        Starting photon energy for the output lineshape grid in **eV**.
    dE : float
        Energy step of the output lineshape grid in **eV**.
    M : int
        Number of energy grid points.
    NN1 : int, optional
        Maximum vibrational quantum number included for the ground state.
        Default 22.
    NN2 : int, optional
        Maximum vibrational quantum number included for the excited state.
        Default 52.

    Attributes
    ----------
    overlap_matrix : numpy.ndarray, shape (NN1+1, NN2+1)
        Franck–Condon overlap integrals ⟨i|j⟩.
    overlap_data : dict
        Keys ``"contributions"`` and ``"energies"`` — per-transition Boltzmann-weighted
        FC squared integrals and their photon energies.
    spectral_data : dict
        Keys ``"energies"``, ``"dos"``, ``"dosw3"`` — the energy grid, Gaussian-broadened
        transition density, and the ω³-weighted (intensity-corrected) lineshape.

    Notes
    -----
    The overlap integral between vibrational levels :math:`i` (ground) and
    :math:`j` (excited) is evaluated recursively using the displaced harmonic
    oscillator recurrence relation.  The full lineshape is then obtained by
    Gaussian broadening with width σ = 0.70·ω\ :sub:`2`.

    The ω³-weighted density ``dosw3`` approximates the spontaneous emission
    rate and is normalized to unit area.

    Examples
    --------
    >>> spec = VibrationalSpectra1D(
    ...     EZPL=1.945, w1_meV=65.0, w2_meV=62.0,
    ...     DQ=1.8, T=300.0, E0=1.3, dE=0.001, M=700
    ... )
    >>> spec.compute_lineshape()
    >>> e_peak, _ = spec.get_peak_position()
    >>> print(f"Peak at {e_peak:.3f} eV,  FWHM = {spec.get_fwhm():.3f} eV")
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
        """Fill ``overlap_matrix`` with Franck–Condon overlaps ⟨i|j⟩ for all (i, j) pairs."""
        for i in range(self.NN1 + 1):
            for j in range(self.NN2 + 1):
                self.overlap_matrix[i, j] = utils.calculate_overlap_element(
                    i, j, self.rho, self.cosfi, self.sinfi
                )

    def compute_spectrum(self) -> None:
        """
        Compute Boltzmann-weighted Franck–Condon transition intensities.

        Populates ``overlap_data`` with per-transition contributions and
        energies, and prints the closure-relation sum (should be ≈ 1).
        """
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
        """
        Gaussian-broaden the FC transitions to produce the PL lineshape.

        Calls :meth:`compute_spectrum` if not already done, then builds the
        energy-resolved transition density ``dos`` and the ω³-weighted
        normalised intensity ``dosw3`` on the grid [E0, E0 + M·dE].
        Results are stored in ``spectral_data``.
        """
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
        """
        Write results to JSON files.

        Parameters
        ----------
        overlap_file : str, optional
            Destination for the MSONable object JSON (includes all inputs).
            Default ``"overlap.json"``.
        lineshape_file : str, optional
            Destination for the ``spectral_data`` dict.
            Default ``"lineshape.json"``.
        """
        Path(overlap_file).write_text(self.to_json(), encoding="utf-8")
        with open(lineshape_file, "w", encoding="utf-8") as f:
            json.dump(self.spectral_data, f, indent=4)

    def plot_lineshape(
        self, save_file: Optional[str] = None, figsize: Tuple[float, float] = (4.4, 4.4)
    ) -> None:
        """
        Plot the normalised PL lineshape (ω³-weighted intensity vs energy).

        Parameters
        ----------
        save_file : str, optional
            If given, save to this path (dpi 600); otherwise ``plt.show()``.
        figsize : tuple of float, optional
            Matplotlib figure size in inches.  Default ``(4.4, 4.4)``.

        Raises
        ------
        ValueError
            If :meth:`compute_lineshape` has not been called yet.
        """
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
        """
        Return the energy and intensity at the lineshape peak.

        Returns
        -------
        (energy, intensity) : tuple of float
            Peak photon energy in eV and the corresponding normalised intensity.

        Raises
        ------
        ValueError
            If :meth:`compute_lineshape` has not been called yet.
        """
        if not self.spectral_data:
            raise ValueError("Run compute_lineshape() before accessing metrics.")
        dosw3 = np.array(self.spectral_data["dosw3"])
        energies = np.array(self.spectral_data["energies"])
        idx_max = np.argmax(dosw3)
        print(f"Peak position: {energies[idx_max]:.3f} eV at {self.T} K.")
        return float(energies[idx_max]), float(dosw3[idx_max])

    def get_fwhm(self) -> float:
        """
        Compute the Full Width at Half Maximum (FWHM) of the PL lineshape.

        Returns
        -------
        float
            FWHM in eV.  Returns 0.0 if fewer than two points exceed half-maximum.

        Raises
        ------
        ValueError
            If :meth:`compute_lineshape` has not been called yet.
        """
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
    Configuration coordinate diagram for a two-state defect transition.

    Manages interpolation of structures between the ground and excited
    equilibrium geometries, DFT input-file generation, potential energy
    surface (PES) extraction from completed Vasprun files, and harmonic
    fitting of the resulting parabolic wells.

    Parameters
    ----------
    ground_struct : pymatgen.core.Structure
        DFT-relaxed ground-state supercell geometry.
    excited_struct : pymatgen.core.Structure
        DFT-relaxed excited-state supercell geometry (same species ordering).

    Attributes
    ----------
    dQ : float
        Mass-weighted configuration coordinate displacement ΔQ in
        amu\\ :sup:`1/2`·Å, computed automatically from the two structures.

    Notes
    -----
    The configuration coordinate is defined as

    .. math::

        \\Delta Q = \\sqrt{\\sum_i m_i\\,|\\Delta\\mathbf{R}_i|^2}

    where the sum runs over all atoms and :math:`\\Delta\\mathbf{R}_i` is
    the displacement of atom *i* between the two equilibrium geometries
    (minimum-image convention with periodic boundary conditions).

    The zero-phonon line energy and Stokes shift can be read from the
    fitted parabola minima returned by :meth:`analyze_ccd`.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> gs = Structure.from_file("CONTCAR_gs")
    >>> es = Structure.from_file("CONTCAR_es")
    >>> ccd = ConfigurationCoordinateDiagram(gs, es)
    >>> print(f"ΔQ = {ccd.dQ:.3f} amu^1/2 Å")
    >>> ccd.setup_calculations(
    ...     displacements=[-0.5, -0.25, 0.25, 0.5, 0.75],
    ...     output_dir="ccd_calcs",
    ...     ground_input_dir="inputs/ground",
    ...     excited_input_dir="inputs/excited",
    ... )
    """

    ground_struct: Structure
    excited_struct: Structure
    dQ: float = field(init=False)

    def __post_init__(self):
        self.dQ = calc_delta_Q(self.ground_struct, self.excited_struct)

    def generate_structures(
        self, displacements: Union[List[float], np.ndarray], remove_zero: bool = True
    ) -> Tuple[List[Structure], List[Structure]]:
        """
        Linearly interpolate structures at fractional displacements along ΔQ.

        Parameters
        ----------
        displacements : array-like of float
            Fractional displacements along the ground→excited path.
            A value of 0.0 is the ground minimum; 1.0 is the excited minimum.
            Values outside [0, 1] extrapolate.
        remove_zero : bool, optional
            Drop any entries equal to 0.0 (ground minimum already exists).
            Default True.

        Returns
        -------
        ground_structs : list of Structure
            Structures displaced from ground toward excited (for ground-state PES).
        excited_structs : list of Structure
            Structures displaced from ground + 1.0 (for excited-state PES).
        """
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
        """
        Write interpolated POSCAR files and copy VASP input files.

        Creates the directory tree::

            output_dir/
              ground/0/POSCAR  KPOINTS  POTCAR  INCAR
              ground/1/...
              excited/0/...
              ...

        Parameters
        ----------
        displacements : array-like of float
            Fractional displacements (see :meth:`generate_structures`).
        output_dir : str or Path
            Root directory for the calculation tree (created if absent).
        ground_input_dir : str or Path
            Directory containing the VASP input files for ground-state runs.
        excited_input_dir : str or Path
            Directory containing the VASP input files for excited-state runs.
        input_files : list of str, optional
            File names to copy from each input directory.
            Default ``["KPOINTS", "POTCAR", "INCAR"]``.
        """
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
        r"""
        Extract (Q, E) data points from a list of completed Vasprun files.

        Parameters
        ----------
        vasprun_paths : list of str or Path
            Paths to ``vasprun.xml`` files, one per interpolated image.
        tol : float, optional
            Fractional-coordinate tolerance for projecting each structure
            onto the configuration coordinate axis.  Default 0.001.

        Returns
        -------
        q_values : numpy.ndarray
            Configuration coordinate values Q in amu\ :sup:`1/2`·Å.
        energies : numpy.ndarray
            Total DFT energies shifted so the minimum is zero (eV).
        """
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
        r"""
        Fit harmonic wells to the ground- and excited-state PES data.

        Parameters
        ----------
        ground_vaspruns : list of str or Path
            ``vasprun.xml`` paths for the ground-state PES images.
        excited_vaspruns : list of str or Path
            ``vasprun.xml`` paths for the excited-state PES images.
        dE : float, optional
            Rigid energy offset applied to the excited-state PES in eV
            (e.g. to set the excited minimum to the ZPL energy).
            Default 0.0.
        plot : bool, optional
            If True, show a matplotlib figure.  Default True.
        figsize : tuple of float, optional
            Figure size in inches.  Default ``(3.3, 3.3)``.
        xlim : tuple of float, optional
            x-axis limits for the CCD plot in amu\ :sup:`1/2`·Å.
        ylim : tuple of float, optional
            y-axis limits for the CCD plot in eV.
        save_plot : str, optional
            If given, save the figure to this path instead of showing.

        Returns
        -------
        (omega_ground, omega_excited) : tuple of float
            Fitted effective angular frequencies in eV for the ground and
            excited harmonic potentials.
        """
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
