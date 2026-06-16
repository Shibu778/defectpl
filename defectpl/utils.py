# -*- coding: utf-8 -*-
"""
Pure mathematical and physics algorithms for defect optical properties calculations.
Author: Shibu Meher, Manoj Dey
"""

import math
from typing import List, Tuple, Union
import numpy as np

from defectpl.constants import AMU2KG, ANG2M, HBAR_JS, HBAR_EVS, EV2J


def calc_delR(dR: np.ndarray) -> float:
    """
    Compute the unweighted Cartesian displacement norm ΔR.

    Parameters
    ----------
    dR : np.ndarray
        Atomic displacement matrix, shape ``(natoms, 3)``, in Å.

    Returns
    -------
    float
        :math:`\\Delta R = \\sqrt{\\sum_{a,i} \\Delta R_{a,i}^2}` in Å.

    Examples
    --------
    >>> import numpy as np
    >>> calc_delR(np.array([[1.0, 2.0, -2.0], [0.0, 0.0, 0.0]]))
    3.0
    """
    return float(np.sqrt(np.sum(dR**2)))


def calc_delQ(masses: np.ndarray, dR: np.ndarray) -> float:
    """
    Compute the mass-weighted configuration coordinate difference ΔQ.

    Parameters
    ----------
    masses : np.ndarray
        Atomic masses in AMU, shape ``(natoms,)``.
    dR : np.ndarray
        Atomic displacement matrix in Å, shape ``(natoms, 3)``.

    Returns
    -------
    float
        :math:`\\Delta Q = \\sqrt{\\sum_a m_a |\\Delta\\mathbf{R}_a|^2}` in
        :math:`\\sqrt{\\text{amu}} \\cdot \\text{Å}`.

    Notes
    -----
    The closure relation :math:`\\Delta Q^2 = \\sum_k q_k^2` holds when the
    phonon eigenvectors span the full displacement space.

    Examples
    --------
    >>> import numpy as np
    >>> masses = np.array([12.011, 15.999])
    >>> dR = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    >>> calc_delQ(masses, dR)  # sqrt(12.011*0.01 + 15.999*0.04)
    0.8289...
    """
    return float(np.sqrt(np.sum(masses * np.sum(dR**2, axis=1))))


def calc_qks(
    masses: np.ndarray, dR: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    """
    Project atomic displacements onto phonon normal modes to find mode coordinates q_k.

    This function uses a loop-based approach to calculate the dimensioned
    configuration coordinates from real-space displacement vectors.

    Parameters
    ----------
    masses : np.ndarray
        1D array of atomic masses for each atom in the system.
        Shape: (N_atoms,). Unit: AMU.
    dR : np.ndarray
        2D array of atomic displacement vectors between two configurations.
        Shape: (N_atoms, 3). Unit: Angstrom (Å).
    eigenvectors : np.ndarray
        3D array representing the normal mode eigenvectors matrix.
        Shape: (N_modes, N_atoms, 3). Dimensionless (normalized).

    Returns
    -------
    np.ndarray
        1D array of projected configuration coordinates for each mode k.
        Shape: (N_modes,). Unit: SI units (kg^{1/2} * m).

    See Also
    --------
    calc_qks_vectorized : Faster, loopless alternative for this calculation.
    """
    qks = []
    sqrt_m = np.sqrt(masses)
    for k in range(len(eigenvectors)):
        qk = np.sum(
            [sqrt_m[i] * np.dot(dR[i], eigenvectors[k][i]) for i in range(len(masses))]
        )
        qk = qk * ANG2M * np.sqrt(AMU2KG)
        qks.append(qk)
    return np.array(qks)


def calc_qks_force_mode(
    masses: np.ndarray,
    forces: np.ndarray,
    eigenvectors: np.ndarray,
    frequencies_eV: np.ndarray,
) -> np.ndarray:
    """
    Project atomic force differences onto phonon normal modes using energies in eV.

    This function calculates the dimensioned configuration coordinates using a
    loop-based approach, derived from the excited state forces acting on the
    ground state geometry. Acoustic or zero-frequency modes are protected to
    prevent division-by-zero errors.

    Parameters
    ----------
    masses : np.ndarray
        1D array of atomic masses for each atom in the system.
        Shape: (N_atoms,). Unit: AMU.
    forces : np.ndarray
        2D array containing the difference in force vectors between the excited
        and ground states. Shape: (N_atoms, 3). Unit: eV/Angstrom (eV/Å).
    eigenvectors : np.ndarray
        3D array representing the normal mode eigenvectors matrix.
        Shape: (N_modes, N_atoms, 3). Dimensionless (normalized).
    frequencies_eV : np.ndarray
        1D array of Gamma-point phonon mode energies.
        Shape: (N_modes,). Unit: Electron-volts (eV).

    Returns
    -------
    np.ndarray
        1D array of projected configuration coordinates for each mode k.
        Shape: (N_modes,). Unit: SI units (kg^{1/2} * m).

    See Also
    --------
    calc_qks_force_vectorized : Faster, loopless alternative for this calculation.
    """
    qks = []
    sqrt_m = np.sqrt(masses)

    for k in range(len(eigenvectors)):
        freq_ev = frequencies_eV[k]

        # Handle the acoustic/zero-frequency modes safely
        if np.isclose(freq_ev, 0.0, atol=1e-5):
            qks.append(0.0)
            continue

        # 1. Raw projection: sum_i (1 / sqrt(M_i)) * dot(F_i, e_{k;i}) -> (eV / Angstrom) / sqrt(AMU)
        proj_sum = np.sum(
            [
                (1.0 / sqrt_m[i]) * np.dot(forces[i], eigenvectors[k][i])
                for i in range(len(masses))
            ]
        )

        # 2. Convert raw projection numerator to SI units -> (Joules / meter) / sqrt(kg)
        proj_sum_SI = proj_sum * (EV2J / ANG2M) / np.sqrt(AMU2KG)

        # 3. Convert energy in eV to angular frequency omega_k (rad/s)
        omega_k = (freq_ev * EV2J) / HBAR_JS

        # 4. Divide by omega_k^2 to find configuration coordinate
        qk_SI = (1.0 / (omega_k**2)) * proj_sum_SI

        qks.append(qk_SI)

    return np.array(qks)


def calc_qks_vectorized(
    masses: np.ndarray, dR: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    """
    Vectorized configuration coordinate calculation from displacements via Einstein summation.

    Eliminates explicit Python loops by using `np.einsum` to project real-space
    atomic displacements onto the full normal mode space simultaneously.

    Parameters
    ----------
    masses : np.ndarray
        1D array of atomic masses for each atom in the system.
        Shape: (N_atoms,). Unit: AMU.
    dR : np.ndarray
        2D array of atomic displacement vectors between two configurations.
        Shape: (N_atoms, 3). Unit: Angstrom (Å).
    eigenvectors : np.ndarray
        3D array representing the normal mode eigenvectors matrix.
        Shape: (N_modes, N_atoms, 3). Dimensionless (normalized).

    Returns
    -------
    np.ndarray
        1D array of projected configuration coordinates for each mode k.
        Shape: (N_modes,). Unit: SI units (kg^{1/2} * m).

    See Also
    --------
    calc_qks : The loop-based alternative for this displacement-based approach.
    """
    # Scale displacements by sqrt(masses)
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    scaled_dR = dR * sqrt_m

    # Project scaled dR onto eigenvectors: sum over atoms (i) and directions (j)
    proj_sum = np.einsum("kij,ij->k", eigenvectors, scaled_dR)

    # Convert unit system from (Angstrom * sqrt(AMU)) to SI (meter * sqrt(kg))
    return proj_sum * ANG2M * np.sqrt(AMU2KG)


def calc_qks_force_vectorized(
    masses: np.ndarray,
    forces: np.ndarray,
    eigenvectors: np.ndarray,
    frequencies_eV: np.ndarray,
) -> np.ndarray:
    """
    Vectorized configuration coordinate calculation from forces using eV energies.

    Utilizes `np.einsum` to bypass explicit loops and projects forces across all
    modes simultaneously. Implements high-throughput masking to safely neutralize
    acoustic and near-zero modes without inducing divide-by-zero exceptions.

    Parameters
    ----------
    masses : np.ndarray
        1D array of atomic masses for each atom in the system.
        Shape: (N_atoms,). Unit: AMU.
    forces : np.ndarray
        2D array containing the difference in force vectors between the excited
        and ground states. Shape: (N_atoms, 3). Unit: eV/Angstrom (eV/Å).
    eigenvectors : np.ndarray
        3D array representing the normal mode eigenvectors matrix.
        Shape: (N_modes, N_atoms, 3). Dimensionless (normalized).
    frequencies_eV : np.ndarray
        1D array of Gamma-point phonon mode energies.
        Shape: (N_modes,). Unit: Electron-volts (eV).

    Returns
    -------
    np.ndarray
        1D array of projected configuration coordinates for each mode k.
        Shape: (N_modes,). Unit: SI units (kg^{1/2} * m).

    See Also
    --------
    calc_qks_force_mode : The loop-based alternative for this force-based approach.
    """
    # 1. Scale forces by 1/sqrt(masses)
    inv_sqrt_m = (1.0 / np.sqrt(masses))[:, np.newaxis]
    scaled_forces = forces * inv_sqrt_m

    # 2. Project onto eigenvectors across all modes simultaneously
    proj_sum = np.einsum("kij,ij->k", eigenvectors, scaled_forces)

    # 3. Convert eV energy values to SI omega squared (rad/s)^2
    omega = (frequencies_eV * EV2J) / HBAR_JS
    omega_sq = omega**2

    # Safely mask zero frequency/acoustic modes
    acoustic_mask = np.isclose(frequencies_eV, 0.0, atol=1e-5)
    omega_sq[acoustic_mask] = np.inf

    # 4. Conversion scalar: converts numerator from [eV / (A * sqrt(AMU))] to SI [J / (m * sqrt(kg))]
    to_SI_numerator = (EV2J / ANG2M) / np.sqrt(AMU2KG)

    # 5. Calculate configuration coordinates
    qks = (proj_sum / omega_sq) * to_SI_numerator
    qks[acoustic_mask] = 0.0

    return qks


def calc_Sks(qks: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """
    Compute partial Huang–Rhys factors for each Gamma-point phonon mode.

    Parameters
    ----------
    qks : np.ndarray
        Mode-projected configuration coordinates in SI units (kg^(1/2)·m),
        shape ``(nmodes,)``.  Obtained from :func:`calc_qks` or
        :func:`calc_qks_force_mode`.
    frequencies : np.ndarray
        Phonon mode energies in eV, shape ``(nmodes,)``.

    Returns
    -------
    np.ndarray
        Partial Huang–Rhys factors :math:`S_k`, dimensionless, shape ``(nmodes,)``.

    Notes
    -----
    The partial HR factor is [Alkauskas 2014a, Eq. 4]:

    .. math::

        S_k = \\frac{\\omega_k q_k^2}{2\\hbar}

    where :math:`q_k` is in SI units (kg^(1/2)·m) and :math:`\\omega_k`
    is converted from eV via :math:`\\omega_k = E_k / \\hbar`.
    The total HR factor is :math:`S = \\sum_k S_k`.
    """
    return frequencies * qks**2 / (2 * HBAR_JS * HBAR_EVS)


def gaussian_broadening(
    omega: Union[float, np.ndarray], omega_k: Union[float, np.ndarray], sigma: float
) -> Union[float, np.ndarray]:
    """
    Evaluate a normalized Gaussian centred at *omega_k* with width *sigma*.

    Used by :func:`calc_S_omega` to replace each discrete delta-function
    mode contribution with a Gaussian of unit area.

    Parameters
    ----------
    omega : float or np.ndarray
        Evaluation energy in eV (can be a grid).
    omega_k : float or np.ndarray
        Centre energy in eV (phonon mode frequency or broadcast array).
    sigma : float
        Standard deviation (broadening width) in eV.

    Returns
    -------
    float or np.ndarray
        Gaussian value(s) at *omega* in eV^{-1}.
    """
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((omega - omega_k) / sigma) ** 2
    )


def calc_S_omega(
    frequencies: np.ndarray,
    Sks: np.ndarray,
    omega_range: List[float],
    sigma: float = 6e-3,
) -> np.ndarray:
    """
    Compute the continuous electron–phonon spectral density function S(ω).

    Each discrete mode contribution :math:`S_k \\delta(\\omega - \\omega_k)` is
    replaced by a normalized Gaussian of width *sigma*, giving:

    .. math::

        S(\\omega) = \\sum_k S_k \\, g(\\omega - \\omega_k, \\sigma)

    Parameters
    ----------
    frequencies : np.ndarray
        Phonon mode energies in eV, shape ``(nmodes,)``.
    Sks : np.ndarray
        Partial Huang–Rhys factors, shape ``(nmodes,)``.
    omega_range : list of float
        Energy grid specification ``[start, stop, npoints]`` in eV.
    sigma : float, default 6e-3
        Gaussian broadening width in eV.  The default 6 meV is suitable for
        defect calculations; increase for broad sidebands.

    Returns
    -------
    np.ndarray
        Spectral density :math:`S(\\omega)` evaluated on the energy grid,
        shape ``(npoints,)``.

    See Also
    --------
    calc_St : Transforms S(ω) to the time domain via inverse FFT.
    """
    omega = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
    S_omega = np.zeros_like(omega)
    for i, w in enumerate(omega):
        S_omega[i] = np.sum(Sks * gaussian_broadening(w, frequencies, sigma))
    return S_omega


def calc_IPR(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Calculate the site-projected Inverse Participation Ratio (IPR) for phonon modes.

    For each normal mode i the per-atom weight is:

        p_{i,a} = sum_{xyz} e_{i,a,xyz}^2

    and the IPR is defined as:

        IPR_i = sum_a p_{i,a}^2 / (sum_a p_{i,a})^2

    Range: 1/N (fully delocalized over N atoms) to 1 (fully localized on one atom).

    Parameters
    ----------
    eigenvectors : np.ndarray
        Phonon normal mode eigenvectors, shape (nmodes, natoms, 3).
        Works for both normalized (phonopy) and un-normalized eigenvectors.

    Returns
    -------
    np.ndarray
        IPR value for each normal mode, shape (nmodes,).
    """
    participations = np.sum(eigenvectors * eigenvectors, axis=2)
    return np.sum(participations**2, axis=1) / np.sum(participations, axis=1) ** 2


def calc_St(S_omega: np.ndarray) -> np.ndarray:
    """
    Transform the electron–phonon spectral density S(ω) to the time domain S(t).

    Parameters
    ----------
    S_omega : np.ndarray
        Spectral density on a uniform energy grid, shape ``(npoints,)``.

    Returns
    -------
    np.ndarray
        Complex-valued time-domain array :math:`S(t)`, shape ``(npoints,)``.

    Notes
    -----
    Implements [Alkauskas 2014a, Eq. 9]:

    .. math::

        S(t) = \\int_0^\\infty S(\\hbar\\omega)\\, e^{-i\\omega t}\\, d(\\hbar\\omega)

    via an inverse FFT with phase centering.

    See Also
    --------
    calc_Gts : Constructs G(t) from S(t).
    """
    Sts = np.fft.ifft(S_omega)
    return 2.0 * np.pi * np.fft.ifftshift(Sts)


def calc_Gts(
    Sts: np.ndarray, total_HR: float, gamma: float, resolution: float
) -> np.ndarray:
    """
    Compute the generating function G(t) from the time-domain spectral function.

    Parameters
    ----------
    Sts : np.ndarray
        Time-domain spectral function S(t), shape ``(npoints,)``.
    total_HR : float
        Total Huang–Rhys factor :math:`S = \\sum_k S_k` (dimensionless).
    gamma : float
        ZPL broadening parameter in meV; applied as a Lorentzian decay
        :math:`e^{-\\gamma|t|}` to reproduce finite ZPL linewidth.
    resolution : float
        Spectral grid density in points per eV (``resolution = npoints / max_energy``).

    Returns
    -------
    np.ndarray
        Complex generating function :math:`G(t)`, shape ``(npoints,)``.

    Notes
    -----
    Implements [Alkauskas 2014a, Eq. 8]:

    .. math::

        G(t) = e^{S(t) - S}\\, e^{-\\gamma|t|}

    The Fourier transform of G(t) gives the optical spectral function A(ℏω).
    """
    n = len(Sts)
    t = (1.0 / resolution) * (np.arange(n) - n / 2)
    return np.exp(Sts - total_HR) * np.exp(-gamma * np.abs(t))


def calc_Spectrum_Intensity(
    Gts: np.ndarray, EZPL: float, resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optical spectral function A(ℏω) and PL intensity L(ℏω) from G(t).

    Parameters
    ----------
    Gts : np.ndarray
        Complex generating function G(t), shape ``(npoints,)``.
    EZPL : float
        Zero-phonon line energy in eV.  Used to shift the spectral axis so that
        the ZPL appears at the correct absolute photon energy.
    resolution : float
        Spectral grid density in points per eV (``npoints / max_energy``).

    Returns
    -------
    A : np.ndarray
        Optical spectral function :math:`A(\\hbar\\omega)`, shape ``(npoints,)``.
        Obtained as the FFT of G(t) with the ZPL shifted to *EZPL*.
    intensity : np.ndarray
        Normalized PL intensity :math:`L(\\hbar\\omega) \\propto \\omega^3 A(\\hbar\\omega)`,
        shape ``(npoints,)``.

    Notes
    -----
    Implements [Alkauskas 2014a, Eq. 7]:

    .. math::

        A(E_{\\text{ZPL}} - \\hbar\\omega) =
        \\frac{1}{2\\pi} \\int_{-\\infty}^{\\infty} G(t)\\, e^{i\\omega t}\\, dt

    The :math:`\\omega^3` prefactor originates from the photon density of states.
    """
    A = np.fft.fft(Gts)
    A1 = A.copy()
    n = len(A)
    shift_idx = int(EZPL * resolution)
    for i in range(n):
        A[(shift_idx - i) % n] = A1[i]
    omega_3 = (np.arange(n) / resolution) ** 3
    return A, A * omega_3


def calculate_hermite(n: int, x: float) -> float:
    """
    Compute the physicist's Hermite polynomial H_n(x) using recurrence relation stability.

    Parameters
    ----------
    n : int
        The order index of the target Hermite polynomial.
    x : float
        The specific coordinate position coordinate value to evaluate.

    Returns
    -------
    float
        The evaluated numerical result of the physicist's Hermite polynomial.
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    y0, y1 = 1.0, 2.0 * x
    for k in range(2, n + 1):
        y0, y1 = y1, 2.0 * x * y1 - 2.0 * (k - 1) * y0
    return y1


def calculate_overlap_element(
    m: int, n: int, rho: float, cosfi: float, sinfi: float
) -> float:
    """
    Calculate the 1D analytical transition overlap integral between two harmonic oscillator states.

    This implements analytical Franck-Condon factor calculations supporting models
    where ground and excited configurations can possess different vibrational frequencies.

    Parameters
    ----------
    m : int
        Vibrational quantum number index mapping the excited state.
    n : int
        Vibrational quantum number index mapping the ground state.
    rho : float
        Dimensionless offset displacement parameter between harmonic minima.
    cosfi : float
        Mixing angle cosine property resolving frequency transformation shifts scaling.
    sinfi : float
        Mixing angle sine property resolving frequency transformation shifts scaling.

    Returns
    -------
    float
        The computed structural overlap integral element value between states (m, n).
    """
    pr1 = (-1) ** n * np.sqrt(2 * cosfi * sinfi) * np.exp(-(rho**2))
    ix = 0.0
    k1, k2 = divmod(m, 2)
    l1, l2 = divmod(n, 2)

    for kx in range(k1 + 1):
        for lx in range(l1 + 1):
            k, lq = 2 * kx + k2, 2 * lx + l2
            try:
                pr2 = (
                    np.sqrt(float(math.factorial(n) * math.factorial(m)))
                    / (
                        math.factorial(k)
                        * math.factorial(lq)
                        * math.factorial(k1 - kx)
                        * math.factorial(l1 - lx)
                    )
                    * 2.0 ** ((k + lq - m - n) // 2)
                )
            except ValueError:
                pr2 = 0
            pr3 = (sinfi**k) * (cosfi**lq)
            f = calculate_hermite(k + lq, rho)
            ix += pr1 * pr2 * pr3 * f

    return ix


def calc_delta_Q(struct1, struct2) -> float:
    """
    Calculate the mass-weighted configuration coordinate displacement delta Q between two structures.

    Parameters
    ----------
    struct1 : pymatgen.core.Structure
        Initial reference equilibrium configuration.
    struct2 : pymatgen.core.Structure
        Final reference equilibrium configuration.

    Returns
    -------
    float
        Mass-weighted structural displacement delta Q in amu^(1/2) * Å.

    Raises
    ------
    ValueError
        If structures have unequal numbers of atoms.
    """
    if len(struct1) != len(struct2):
        raise ValueError("Structures must have the same number of atoms.")

    from pymatgen.util.coord import pbc_shortest_vectors

    masses = np.array([site.specie.atomic_mass for site in struct1.sites])
    lattice = struct1.lattice
    n = len(struct1)
    dR = np.vstack(
        [
            pbc_shortest_vectors(
                lattice, struct1.frac_coords[i], struct2.frac_coords[i]
            )
            for i in range(n)
        ]
    ).reshape(n, 3)
    return calc_delQ(masses, dR)


def get_omega_from_pes(
    q_values: np.ndarray,
    energies: np.ndarray,
    ax=None,
    eval_grid: np.ndarray = None,
) -> float:
    """
    Fit a harmonic parabola E(Q) = ½·M·ω²·Q² to PES data points.

    A second-order polynomial is fit to *(Q, E)* data.  The effective angular
    frequency ω is extracted from the curvature coefficient and returned in
    **eV** (as an effective phonon energy ℏω).

    Parameters
    ----------
    q_values : np.ndarray
        Configuration coordinate values Q in amu^(1/2)·Å.
    energies : np.ndarray
        Total energies (or energy differences) in eV, zero-referenced to the minimum.
    ax : matplotlib.axes.Axes, optional
        If provided, the fitted parabola is plotted on this axes object.
    eval_grid : np.ndarray, optional
        Q grid for plotting the fit curve.  Required when *ax* is given.

    Returns
    -------
    float
        Effective phonon energy ℏω in **eV** derived from the parabola curvature.

    Notes
    -----
    The fit uses :func:`numpy.polyfit` (degree 2).  The curvature coefficient
    a₂ (eV/amu·Å²) is related to the angular frequency by:

    .. math::

        \\hbar\\omega = \\sqrt{2 a_2 / M_{\\text{ref}}}

    where an effective reference mass of 1 amu is assumed (since Q is already
    mass-weighted).  The units are handled via the AMU2KG, ANG2M, and EV2J
    conversion factors.
    """
    coeffs = np.polyfit(q_values, energies, 2)
    a2 = coeffs[0]  # coefficient of Q^2 in eV / (amu^1/2 * Å)^2

    # Convert curvature to SI: [eV / (amu * Å^2)] → [J / (kg * m^2)] = [rad^2/s^2]
    a2_SI = a2 * EV2J / (AMU2KG * ANG2M**2)
    omega_SI = np.sqrt(max(2.0 * a2_SI, 0.0))  # rad/s
    omega_eV = omega_SI * HBAR_JS / EV2J

    if ax is not None and eval_grid is not None:
        poly_fn = np.poly1d(coeffs)
        ax.plot(eval_grid, poly_fn(eval_grid), "--", lw=1)

    return float(omega_eV)


def extract_important_properties(
    pl_engine, filename: str = "important_properties.txt"
) -> None:
    """
    Write a structured summary of scalar PL properties to a text file.

    Extracts S, DW factor, ΔQ, ΔR, and configuration metadata from a
    :class:`~defectpl.defectpl.Photoluminescence` instance.  Force-mode
    runs omit ΔQ/ΔR since they are not directly computed in that mode.

    Parameters
    ----------
    pl_engine : Photoluminescence
        A fully computed Photoluminescence instance.
    filename : str, optional
        Output file path.  Default ``"important_properties.txt"``.
    """
    import numpy as np

    # 1. Determine the calculation mode based on input state flags
    is_force_mode = (
        hasattr(pl_engine, "dF") and pl_engine.dF is not None and np.any(pl_engine.dF)
    )
    calc_mode = "Force Mode" if is_force_mode else "Displacement Mode"

    # 2. Build the structural layout header
    lines = [
        "===================================================",
        "                PROPERTIES SUMMARY                 ",
        "===================================================",
        f"Calculation Run Mode         : {calc_mode:>12}",
        f"Zero-Phonon Line (ZPL) Energy : {getattr(pl_engine, 'EZPL', 'N/A'):>12} eV",
        f"Total Huang-Rhys (HR) Factor : {getattr(pl_engine, 'HR_factor', 'N/A'):>12.6f}",
        f"Debye-Waller (DW) Factor     : {getattr(pl_engine, 'DW_factor', 'N/A'):>12.6f}",
    ]

    # 3. Conditional Step: Only inject spatial metrics if we are NOT in Force Mode
    if not is_force_mode:
        lines.extend(
            [
                f"Mass-Weighted Delta Q (delQ)  : {getattr(pl_engine, 'delQ', 'N/A'):>12.6f} amu^(1/2)*Å",
                f"Structural Delta R (delR)     : {getattr(pl_engine, 'delR', 'N/A'):>12.6f} Å",
            ]
        )

    # 4. Append operational configuration tracking metadata
    lines.extend(
        [
            "---------------------------------------------------",
            f"Total Number of Atoms (natoms): {getattr(pl_engine, 'natoms', 'N/A'):>12}",
            f"ZPL Broadening Factor (gamma) : {getattr(pl_engine, 'gamma', 'N/A'):>12.2f} meV",
            f"Gaussian Broadening (sigma)   : {getattr(pl_engine, 'sigma', 'N/A'):>12.6f} eV",
            f"Energy Mesh Resolution        : {getattr(pl_engine, 'resolution', 'N/A'):>12} points/eV",
            "===================================================",
        ]
    )

    # Write cleanly to file out
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Important parameters successfully exported to text summary: {filename}")
