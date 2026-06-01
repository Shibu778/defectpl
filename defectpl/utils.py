# -*- coding: utf-8 -*-
"""
Pure mathematical and physics algorithms for defect optical properties calculations.
Author: Shibu Meher, Manoj Dey
"""

import math
from typing import List, Tuple, Union
import numpy as np
from pymatgen.core import Structure
from pymatgen.util.coord import pbc_shortest_vectors

from defectpl.constants import AMU2KG, ANG2M, HBAR_JS, HBAR_EVS


def calc_delR(dR: np.ndarray) -> float:
    """
    Calculate the global norm of coordinate differences.

    Parameters
    ----------
    dR : np.ndarray
        Array containing displacement vectors.

    Returns
    -------
    float
        The root-sum-squared global structural displacement.
    """
    return float(np.sqrt(np.sum(dR**2)))


def calc_delQ(masses: np.ndarray, dR: np.ndarray) -> float:
    """
    Calculate the mass-weighted generalized configuration coordinate displacement delta Q.

    Parameters
    ----------
    masses : np.ndarray
        Array of atomic masses.
    dR : np.ndarray
        Array containing displacement vectors for each site.

    Returns
    -------
    float
        The mass-weighted configuration coordinate displacement.
    """
    return float(np.sqrt(np.sum(masses * np.sum(dR**2, axis=1))))


def calc_qks(masses: np.ndarray, dR: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """
    Project atomic displacements onto phonon normal modes to find mode coordinates q_k.

    Parameters
    ----------
    masses : np.ndarray
        Array of atomic masses.
    dR : np.ndarray
        Array containing atomic displacement vectors.
    eigenvectors : np.ndarray
        Phonon normal mode eigenvectors matrix.

    Returns
    -------
    np.ndarray
        The projected configuration displacements array for each mode k.
    """
    qks = []
    for k in range(len(eigenvectors)):
        qk = 0.0
        for i in range(len(masses)):
            qk += np.sqrt(masses[i]) * np.dot(dR[i], eigenvectors[k][i])
        qk = qk * ANG2M * np.sqrt(AMU2KG)
        qks.append(qk)
    return np.array(qks)


def calc_Sks(qks: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """
    Calculate partial Huang-Rhys (HR) factors for each phonon normal mode k.

    Parameters
    ----------
    qks : np.ndarray
        Normal mode projection coordinates array.
    frequencies : np.ndarray
        Phonon frequencies array.

    Returns
    -------
    np.ndarray
        The calculated partial Huang-Rhys factors per mode.
    """
    return frequencies * qks**2 / (2 * HBAR_JS * HBAR_EVS)


def gaussian_broadening(
    omega: Union[float, np.ndarray], omega_k: Union[float, np.ndarray], sigma: float
) -> Union[float, np.ndarray]:
    """
    Evaluate a standard normalized Gaussian profile for spectral broadening.

    Parameters
    ----------
    omega : float or np.ndarray
        Energy sampling points axis grid.
    omega_k : float or np.ndarray
        Central resonance energy peaks grid positions.
    sigma : float
        Standard deviation width parameter managing Gaussian attenuation.

    Returns
    -------
    float or np.ndarray
        The evaluated intensity of the Gaussian curve profile.
    """
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((omega - omega_k) / sigma) ** 2)


def calc_S_omega(
    frequencies: np.ndarray, Sks: np.ndarray, omega_range: List[float], sigma: float = 6e-3
) -> np.ndarray:
    """
    Compute the continuous Huang-Rhys spectral density function S(omega) using broadening.

    Parameters
    ----------
    frequencies : np.ndarray
        Phonon normal mode frequencies array.
    Sks : np.ndarray
        Partial Huang-Rhys factors array.
    omega_range : list of float
        Grid array generation specification boundary parameters: [start, stop, num_points].
    sigma : float, default 6e-3
        Gaussian broadening standard deviation mapping width.

    Returns
    -------
    np.ndarray
        The continuous spectral density array function evaluated over the energy grid.
    """
    omega = np.linspace(omega_range[0], omega_range[1], int(omega_range[2]))
    S_omega = np.zeros_like(omega)
    for i, w in enumerate(omega):
        S_omega[i] = np.sum(Sks * gaussian_broadening(w, frequencies, sigma))
    return S_omega


def calc_IPR(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Calculate the Inverse Participation Ratio (IPR) to evaluate localization of modes.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Phonon normal mode eigenvectors matrix.

    Returns
    -------
    np.ndarray
        The inverse participation ratio array for each normal mode.
    """
    participations = np.sum(eigenvectors * eigenvectors, axis=2)
    return 1.0 / np.sum(participations**2, axis=1)


def calc_St(S_omega: np.ndarray) -> np.ndarray:
    """
    Transform the spectral density function S(omega) to the time domain S(t) via inverse FFT.

    Parameters
    ----------
    S_omega : np.ndarray
        Continuous frequency-domain spectral density array.

    Returns
    -------
    np.ndarray
        The corresponding complex-valued time domain array function S(t).
    """
    Sts = np.fft.ifft(S_omega)
    return 2.0 * np.pi * np.fft.ifftshift(Sts)


def calc_Gts(Sts: np.ndarray, total_HR: float, gamma: float, resolution: float) -> np.ndarray:
    """
    Compute the generating function G(t) tracking time-dependent correlations.

    Parameters
    ----------
    Sts : np.ndarray
        Time domain transformed array function tracker S(t).
    total_HR : float
        Summed total global Huang-Rhys factor scalar component.
    gamma : float
        Damping factor managing phenomenological electronic decay lifetime scaling.
    resolution : float
        Sampling frequency or grid interval inverse step scaling ratio.

    Returns
    -------
    np.ndarray
        The resulting multi-mode generating function time array trajectory G(t).
    """
    l = len(Sts)
    t = (1.0 / resolution) * (np.arange(l) - l / 2)
    return np.exp(Sts - total_HR) * np.exp(-gamma * np.abs(t))


def calc_Spectrum_Intensity(Gts: np.ndarray, EZPL: float, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the generating function back to frequency space to get final line intensities.

    Parameters
    ----------
    Gts : np.ndarray
        The time domain generating function array tracker G(t).
    EZPL : float
        Zero-phonon line transition energy value boundary scalar (eV).
    resolution : float
        Grid array frequency sampling accuracy ratio configuration.

    Returns
    -------
    luminescence_dos : np.ndarray
        Raw line shape density of states function array matrix (A).
    luminescence_intensity : np.ndarray
        Luminescence intensity array spectrum scaled by omega^3 energy modulation.
    """
    A = np.fft.fft(Gts)
    A1 = A.copy()
    l = len(A)
    shift_idx = int(EZPL * resolution)
    for i in range(l):
        A[(shift_idx - i) % l] = A1[i]
    omega_3 = (np.arange(l) / resolution) ** 3
    return A, A * omega_3


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

    frac_coords1 = struct1.frac_coords
    frac_coords2 = struct2.frac_coords
    length = len(frac_coords1)

    dR = pbc_shortest_vectors(struct1.lattice, frac_coords1, frac_coords2)
    dR = np.diagonal(dR, axis1=0, axis2=1).T
    squared_displacements = np.sum(dR**2, axis=1)

    return float(np.sqrt(np.sum(masses * squared_displacements)))


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
            k, l = 2 * kx + k2, 2 * lx + l2
            try:
                pr2 = (
                    np.sqrt(float(math.factorial(n) * math.factorial(m)))
                    / (
                        math.factorial(k)
                        * math.factorial(l)
                        * math.factorial(k1 - kx)
                        * math.factorial(l1 - lx)
                    )
                    * 2.0 ** ((k + l - m - n) // 2)
                )
            except ValueError:
                pr2 = 0
            pr3 = (sinfi**k) * (cosfi**l)
            f = calculate_hermite(k + l, rho)
            ix += pr1 * pr2 * pr3 * f

    return ix