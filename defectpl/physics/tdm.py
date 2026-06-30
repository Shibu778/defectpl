# -*- coding: utf-8 -*-
"""
defectpl.physics.tdm
====================
Transition Dipole Moment (TDM) calculations from VASP WAVECAR / vaspwave.h5,
plus Inverse Participation Ratio (IPR) of Kohn-Sham states and optical
properties (ZPL, Einstein A coefficient, radiative lifetime).

Classes
-------
WavecarReader
    Standalone binary WAVECAR reader (numpy / scipy only; no pymatgen).
    Supports standard, Gamma-only, and SOC calculations.
    Supports compact WAVECAR format (metadata embedded in header).
VaspwaveH5Reader
    Reads vaspwave.h5 (VASP 6 HDF5 output).  Requires h5py.

Free functions
--------------
select_bands          — Band index selection helper (near-Fermi, energy window, …)
compute_ipr_band      — IPR, PR and normalised PR for one KS state
compute_ipr_all       — IPR / PR for all selected bands at all k-points
compute_ipr_weighted  — k-weight-averaged IPR for one band
save_ipr_json         — Serialise :func:`compute_ipr_all` result to JSON
save_ipr_csv          — Serialise band-level IPR summary to CSV
get_zpl               — Zero-phonon line from two VASP directories
get_dQ                — Mass-weighted config-coordinate shift dQ
get_einstein_coefficient — Einstein A (MHz) from ZPL + TDM + refractive index
get_radiative_lifetime   — Lifetime (ns) from Einstein A
compute_optical_properties — Full optical property workflow

Physical constants match VASP internal values so TDMs are numerically
compatible with VASP-computed matrix elements.

References
----------
* WAVECAR format: http://www.andrew.cmu.edu/user/feenstra/wavetrans/
* TDM in periodic systems: Phys. Rev. B 87, 125301 (2013)
* Alkauskas et al., Phys. Rev. B 90, 075202 (2014)  — IPR definition
"""

from __future__ import annotations

import csv
import json
import os
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (matching VASP internal values)
# ---------------------------------------------------------------------------

AUTOA = 0.529177249  # 1 Bohr in Angstrom
RYTOEV = 13.605826  # 1 Ry in eV
AUTDEBYE = 2.541746  # 1 a.u. dipole in Debye
TPI = 2.0 * np.pi  # 2π
HSQDTM = RYTOEV * AUTOA * AUTOA  # ℏ²/(2mₑ) in eV·Å²

__all__ = [
    # Readers
    "WavecarReader",
    "VaspwaveH5Reader",
    # Band selection
    "select_bands",
    # IPR
    "compute_ipr_band",
    "compute_ipr_all",
    "compute_ipr_weighted",
    "save_ipr_json",
    "save_ipr_csv",
    # Optical properties
    "get_zpl",
    "get_dQ",
    "get_einstein_coefficient",
    "get_radiative_lifetime",
    "compute_optical_properties",
    # Module-level helpers
    "get_weighted_avg_tdm",
    "read_ibzkpt_weights",
]


# ---------------------------------------------------------------------------
# Module-level utility helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def read_ibzkpt_weights(ibzkpt_file: Union[str, Path]) -> np.ndarray:
    """Read irreducible k-point weights from an IBZKPT file.

    Delegates to :func:`defectpl.io.wavecar.read_ibzkpt_weights`; provided
    here for convenience so callers do not need a second import.

    Parameters
    ----------
    ibzkpt_file : str or Path

    Returns
    -------
    np.ndarray, shape (nkpts,)
        Raw (unnormalised) weights.
    """
    from defectpl.io.wavecar import read_ibzkpt_weights as _rio

    return _rio(ibzkpt_file)


def get_weighted_avg_tdm(
    tdm_per_kpt: np.ndarray,
    kweights: np.ndarray,
) -> np.ndarray:
    """Compute the k-point weighted average of TDM components.

    Parameters
    ----------
    tdm_per_kpt : np.ndarray, shape (nkpts, 3)
        Absolute TDM vector at each k-point.
    kweights : np.ndarray, shape (nkpts,)
        Raw k-point weights from IBZKPT.

    Returns
    -------
    np.ndarray, shape (3,)
        Weighted-average |TDM| components in Debye.
    """
    w = np.asarray(kweights, dtype=float)
    w = w / w.sum()
    return np.einsum("k,kc->c", w, tdm_per_kpt)


# ---------------------------------------------------------------------------
# Internal record-write helper
# ---------------------------------------------------------------------------


def _write_record(fout, data: np.ndarray, recl: int) -> None:
    """Write a zero-padded record of exactly ``recl`` bytes."""
    raw = data.tobytes()
    padded = raw + b"\x00" * (recl - len(raw))
    fout.write(padded[:recl])


# ---------------------------------------------------------------------------
# Band-pair selection at a single k-point (used by get_all_transitions)
# ---------------------------------------------------------------------------


def _select_band_pairs_at_kpoint(
    bands_arr: np.ndarray,
    occs_arr: np.ndarray,
    ispin: int,
    ikpt: int,
    mode: str,
    occ_bands: Optional[Tuple[int, int]] = None,
    unocc_bands: Optional[Tuple[int, int]] = None,
    occ_band_list: Optional[Sequence[int]] = None,
    unocc_band_list: Optional[Sequence[int]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    fermi_level: float = 0.0,
    occ_threshold: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Return ``(occ_list, unocc_list)`` band indices for one k-point."""
    s = ispin - 1
    k = ikpt - 1
    nbands = bands_arr.shape[2]

    if mode == "band_range":
        occ_list = list(range(occ_bands[0], occ_bands[1] + 1))
        unocc_list = list(range(unocc_bands[0], unocc_bands[1] + 1))

    elif mode == "band_list":
        occ_list = [int(b) for b in occ_band_list]
        unocc_list = [int(b) for b in unocc_band_list]

    elif mode == "energy":
        E_lo, E_hi = energy_range
        occ_list, unocc_list = [], []
        for b in range(1, nbands + 1):
            E_rel = bands_arr[s, k, b - 1] - fermi_level
            if E_lo <= E_rel <= E_hi:
                if E_rel <= 0.0:
                    occ_list.append(b)
                else:
                    unocc_list.append(b)

    elif mode == "occupation":
        occ_list, unocc_list = [], []
        for b in range(1, nbands + 1):
            if occs_arr[s, k, b - 1] >= occ_threshold:
                occ_list.append(b)
            else:
                unocc_list.append(b)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Choose from 'occupation', 'energy', 'band_range', 'band_list'."
        )

    return occ_list, unocc_list


# ===========================================================================
# WavecarReader — standalone binary WAVECAR reader
# ===========================================================================


class WavecarReader:
    """Read VASP pseudo-wavefunctions from a WAVECAR binary file.

    Implements the WAVECAR format independently of pymatgen.  Supports
    standard, Gamma-only (``lgamma``), and non-collinear / SOC (``lsorbit``)
    calculations.  Also supports a *compact* WAVECAR format (see
    :meth:`save_compact_wavecar`) in which original band indices and
    ``nbands`` are embedded in the file header — no companion file is needed.

    Parameters
    ----------
    wavecar : str or Path
        Path to the WAVECAR file.  Compressed files (``.gz``, ``.bz2``,
        ``.xz``) are handled transparently via
        :func:`~defectpl.io.wavecar.open_wavecar`.
    lgamma : bool, optional
        ``True`` for Gamma-point-only VASP executables.  Default: ``False``.
    lsorbit : bool, optional
        ``True`` for non-collinear (SOC) calculations.  Default: ``False``.
    gamma_half : {'x', 'z'}, optional
        Which half of the FFT grid VASP stores in the Gamma-only case.
        Default: ``'x'`` (VASP ≥ 5.4).

    Attributes
    ----------
    nkpts : int
    nbands : int
    nspin : int
    encut : float
    Acell : np.ndarray, shape (3, 3)   — real-space lattice in Å
    Bcell : np.ndarray, shape (3, 3)   — reciprocal lattice in Å⁻¹ (no 2π)
    kvecs : np.ndarray, shape (nkpts, 3)
    bands : np.ndarray, shape (nspin, nkpts, nbands)  — eigenvalues in eV
    occs  : np.ndarray, shape (nspin, nkpts, nbands)  — occupancies

    Examples
    --------
    >>> wfc = WavecarReader("WAVECAR")
    >>> result = wfc.get_tdm_all_kpoints(ispin=1, iband_i=638, iband_j=639)
    >>> print(result["tdm_magnitude"])
    """

    _COMPACT_MARKER: float = 1.0e38

    def __init__(
        self,
        wavecar: Union[str, Path] = "WAVECAR",
        lgamma: bool = False,
        lsorbit: bool = False,
        gamma_half: str = "x",
    ) -> None:
        self._fname = Path(wavecar)
        self._lgam = lgamma
        self._lsoc = lsorbit
        self._gam_half = gamma_half.lower()

        if lsorbit and lgamma:
            raise ValueError("lsorbit and lgamma are mutually exclusive.")
        if self._gam_half not in ("x", "z"):
            raise ValueError("gamma_half must be 'x' or 'z'.")

        try:
            from defectpl.io.wavecar import open_wavecar

            self._fh, self._tmp_path = open_wavecar(str(wavecar))
        except Exception:
            self._fh = open(self._fname, "rb")
            self._tmp_path = None

        self._read_header()
        self._read_bands()

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass
        if getattr(self, "_tmp_path", None):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Header / band reading
    # ------------------------------------------------------------------

    def _read_header(self) -> None:
        self._fh.seek(0)
        rec = np.fromfile(self._fh, dtype=np.float64, count=3)
        self._recl = int(rec[0])
        self.nspin = int(rec[1])
        self._rtag = int(rec[2])
        self._WFPrec = self._set_precision()

        self._fh.seek(self._recl)
        rec2 = np.fromfile(self._fh, dtype=np.float64, count=12)
        self.nkpts = int(rec2[0])
        self.nbands = int(rec2[1])
        self.encut = rec2[2]
        self.Acell = rec2[3:].reshape((3, 3))
        self.Omega = np.linalg.det(self.Acell)
        self.Bcell = np.linalg.inv(self.Acell).T

        Anorm = np.linalg.norm(self.Acell, axis=1)
        cutof = np.ceil(sqrt(self.encut / RYTOEV) / (TPI / (Anorm / AUTOA)))
        self._ngrid = np.array(2 * cutof + 1, dtype=int)

    def _set_precision(self) -> type:
        tags = {45200: np.complex64, 45210: np.complex128}
        if self._rtag not in tags:
            raise ValueError(f"Unknown WAVECAR precision TAG: {self._rtag}")
        return tags[self._rtag]

    def _read_bands(self) -> None:
        self._nplws = np.zeros(self.nkpts, dtype=int)
        self.kvecs = np.zeros((self.nkpts, 3), dtype=float)
        self.bands = np.zeros((self.nspin, self.nkpts, self.nbands), dtype=float)
        self.occs = np.zeros((self.nspin, self.nkpts, self.nbands), dtype=float)

        _compact_detected = False
        _compact_band_map_raw: List[int] = []
        _compact_original_nbands_raw: int = 0

        for ispin in range(self.nspin):
            for ikpt in range(self.nkpts):
                rec = self._where_rec(ispin + 1, ikpt + 1, 1) - 1
                self._fh.seek(rec * self._recl)
                dump = np.fromfile(
                    self._fh, dtype=np.float64, count=4 + 3 * self.nbands
                )
                if ispin == 0:
                    self._nplws[ikpt] = int(dump[0])
                    self.kvecs[ikpt] = dump[1:4]
                data = dump[4:].reshape(-1, 3)
                self.bands[ispin, ikpt, :] = data[:, 0]
                self.occs[ispin, ikpt, :] = data[:, 2]

                if ispin == 0 and ikpt == 0:
                    extra = np.fromfile(self._fh, dtype=np.float64, count=2)
                    if (
                        len(extra) == 2
                        and abs(extra[0] - self._COMPACT_MARKER) < 1.0
                        and extra[1] >= 1.0
                    ):
                        _compact_detected = True
                        _compact_band_map_raw = [int(round(x)) for x in data[:, 1]]
                        _compact_original_nbands_raw = int(round(extra[1]))

        if _compact_detected:
            self._compact_band_map: List[int] = _compact_band_map_raw
            self._compact_file_nbands: int = self.nbands
            original_nbands = _compact_original_nbands_raw
            full_bands = np.full((self.nspin, self.nkpts, original_nbands), np.nan)
            full_occs = np.full((self.nspin, self.nkpts, original_nbands), np.nan)
            for i, orig_b in enumerate(_compact_band_map_raw):
                full_bands[:, :, orig_b - 1] = self.bands[:, :, i]
                full_occs[:, :, orig_b - 1] = self.occs[:, :, i]
            self.bands = full_bands
            self.occs = full_occs
            self.nbands = original_nbands

    def _where_rec(self, ispin: int, ikpt: int, iband: int) -> int:
        file_nbands = getattr(self, "_compact_file_nbands", self.nbands)
        return (
            2
            + (ispin - 1) * self.nkpts * (file_nbands + 1)
            + (ikpt - 1) * (file_nbands + 1)
            + iband
        )

    def _check_index(self, ispin: int, ikpt: int, iband: int) -> None:
        if not (1 <= ispin <= self.nspin):
            raise IndexError(f"ispin={ispin} out of range [1, {self.nspin}].")
        if not (1 <= ikpt <= self.nkpts):
            raise IndexError(f"ikpt={ikpt} out of range [1, {self.nkpts}].")
        if not (1 <= iband <= self.nbands):
            raise IndexError(f"iband={iband} out of range [1, {self.nbands}].")

    # ------------------------------------------------------------------
    # G-vector generation
    # ------------------------------------------------------------------

    def gvectors(self, ikpt: int = 1) -> np.ndarray:
        """G-vectors satisfying ``|G + k|² / 2 < ENCUT``.

        Parameters
        ----------
        ikpt : int
            k-point index (1-based).

        Returns
        -------
        np.ndarray, shape (nplw, 3), dtype int
            Integer Miller indices.
        """
        if not (1 <= ikpt <= self.nkpts):
            raise IndexError(f"ikpt={ikpt} out of range [1, {self.nkpts}].")

        kvec = self.kvecs[ikpt - 1]
        lgam = self._lgam

        fx = np.arange(self._ngrid[0], dtype=int)
        fy = np.arange(self._ngrid[1], dtype=int)
        fz = np.arange(self._ngrid[2], dtype=int)
        fx[self._ngrid[0] // 2 + 1 :] -= self._ngrid[0]
        fy[self._ngrid[1] // 2 + 1 :] -= self._ngrid[1]
        fz[self._ngrid[2] // 2 + 1 :] -= self._ngrid[2]

        if lgam:
            if self._gam_half == "x":
                fx = fx[: self._ngrid[0] // 2 + 1]
            else:
                fz = fz[: self._ngrid[2] // 2 + 1]

        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape(3, -1)
        kgrid = np.column_stack([gx, gy, gz]).astype(float)

        if lgam:
            if self._gam_half == "z":
                mask = (
                    (gz > 0)
                    | ((gz == 0) & (gy > 0))
                    | ((gz == 0) & (gy == 0) & (gx >= 0))
                )
            else:
                mask = (
                    (gx > 0)
                    | ((gx == 0) & (gy > 0))
                    | ((gx == 0) & (gy == 0) & (gz >= 0))
                )
            kgrid = kgrid[mask]

        ke = (
            HSQDTM
            * np.linalg.norm(
                np.dot(kgrid + kvec[np.newaxis, :], TPI * self.Bcell), axis=1
            )
            ** 2
        )
        return np.asarray(kgrid[ke < self.encut], dtype=int)

    # ------------------------------------------------------------------
    # Coefficient reading
    # ------------------------------------------------------------------

    def read_band_coeff(
        self,
        ispin: int = 1,
        ikpt: int = 1,
        iband: int = 1,
        norm: bool = False,
    ) -> np.ndarray:
        """Read plane-wave coefficients for a single KS state.

        Parameters
        ----------
        ispin : int  — spin index (1-based)
        ikpt  : int  — k-point index (1-based)
        iband : int  — band index (1-based, original numbering)
        norm  : bool — if True, normalise to unit norm

        Returns
        -------
        np.ndarray, shape (nplw,), dtype complex128
        """
        self._check_index(ispin, ikpt, iband)
        if hasattr(self, "_compact_band_map") and self._compact_band_map:
            cmap = self._compact_band_map
            if iband not in cmap:
                raise KeyError(
                    f"Band {iband} not stored in compact WAVECAR. "
                    f"Available: {sorted(cmap)}"
                )
            iband = cmap.index(iband) + 1

        rec = self._where_rec(ispin, ikpt, iband)
        self._fh.seek(rec * self._recl)
        nplw = self._nplws[ikpt - 1]
        raw = np.fromfile(self._fh, dtype=self._WFPrec, count=nplw)
        cg = raw.astype(np.complex128)
        if norm:
            cg /= np.linalg.norm(cg)
        return cg

    # ------------------------------------------------------------------
    # Momentum and dipole matrix elements
    # ------------------------------------------------------------------

    def get_momentum_matrix(
        self,
        ispin: int,
        ikpt: int,
        iband_i: int,
        iband_j: int,
        cg_i: Optional[np.ndarray] = None,
        cg_j: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute ``<psi_j | p | psi_i>`` in the plane-wave basis.

        Returns
        -------
        np.ndarray, shape (3,), dtype complex128
            Momentum matrix element (x, y, z) in atomic units (ℏ/Å).
        """
        self._check_index(ispin, ikpt, iband_i)
        self._check_index(ispin, ikpt, iband_j)

        k0 = self.kvecs[ikpt - 1]
        G0 = self.gvectors(ikpt)
        Gk = np.dot(G0 + k0, self.Bcell * TPI)

        CG_i = cg_i if cg_i is not None else self.read_band_coeff(ispin, ikpt, iband_i)
        CG_j = cg_j if cg_j is not None else self.read_band_coeff(ispin, ikpt, iband_j)
        ovlap = CG_j.conj() * CG_i

        if self._lgam:
            mom = np.sum(ovlap[:, None] * Gk, axis=0)
            mom -= np.sum(ovlap[:, None].conj() * Gk, axis=0)
            mom /= 2.0
        elif self._lsoc:
            mom = np.sum(ovlap[:, None] * np.vstack([Gk, Gk]), axis=0)
        else:
            mom = np.sum(ovlap[:, None] * Gk, axis=0)

        return mom

    def get_dipole_matrix(
        self,
        ispin: int,
        ikpt: int,
        iband_i: int,
        iband_j: int,
        cg_i: Optional[np.ndarray] = None,
        cg_j: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Electric dipole matrix element (transition dipole moment).

        Converts the momentum matrix element to the length-gauge dipole via
        the p–r relation::

            <psi_j|r|psi_i> = -i / (ΔE / 2Ry) × <psi_j|p|psi_i> × a₀ × D_conv

        Returns
        -------
        E_i : float — eV
        E_j : float — eV
        dE  : float — eV
        dipole : np.ndarray, shape (3,), complex128 — Debye
        """
        self._check_index(ispin, ikpt, iband_i)
        self._check_index(ispin, ikpt, iband_j)

        Ei = self.bands[ispin - 1, ikpt - 1, iband_i - 1]
        Ej = self.bands[ispin - 1, ikpt - 1, iband_j - 1]
        dE = Ej - Ei

        if np.isclose(dE, 0.0):
            return Ei, Ej, dE, np.zeros(3, dtype=complex)

        mom = self.get_momentum_matrix(ispin, ikpt, iband_i, iband_j, cg_i, cg_j)
        dipole = -1j / (dE / (2.0 * RYTOEV)) * mom * AUTOA * AUTDEBYE
        return Ei, Ej, dE, dipole

    # ------------------------------------------------------------------
    # All-k-point TDM
    # ------------------------------------------------------------------

    def get_tdm_all_kpoints(
        self,
        ispin: int,
        iband_i: int,
        iband_j: int,
    ) -> dict:
        """TDM between two bands at every k-point.

        Parameters
        ----------
        ispin : int — spin channel (1-based)
        iband_i : int — initial band (1-based)
        iband_j : int — final band (1-based)

        Returns
        -------
        dict with keys:
            ``"ispin"``, ``"iband_i"``, ``"iband_j"``,
            ``"kvecs"`` shape (nkpts, 3),
            ``"E_i"`` / ``"E_j"`` / ``"dE"`` shape (nkpts,),
            ``"tdm_components"`` shape (nkpts, 3) — |TDM| in Debye,
            ``"tdm_magnitude"`` shape (nkpts,) — Euclidean norm in Debye.

        Examples
        --------
        >>> wfc = WavecarReader("WAVECAR")
        >>> res = wfc.get_tdm_all_kpoints(1, 638, 639)
        """
        Ei_arr = np.zeros(self.nkpts)
        Ej_arr = np.zeros(self.nkpts)
        dE_arr = np.zeros(self.nkpts)
        tdm_comp = np.zeros((self.nkpts, 3))

        for ikpt in range(1, self.nkpts + 1):
            Ei, Ej, dE, dip = self.get_dipole_matrix(ispin, ikpt, iband_i, iband_j)
            Ei_arr[ikpt - 1] = Ei
            Ej_arr[ikpt - 1] = Ej
            dE_arr[ikpt - 1] = dE
            tdm_comp[ikpt - 1] = np.abs(dip)

        return {
            "ispin": ispin,
            "iband_i": iband_i,
            "iband_j": iband_j,
            "kvecs": self.kvecs.copy(),
            "E_i": Ei_arr,
            "E_j": Ej_arr,
            "dE": dE_arr,
            "tdm_components": tdm_comp,
            "tdm_magnitude": np.linalg.norm(tdm_comp, axis=1),
        }

    # ------------------------------------------------------------------
    # Weighted-average TDM
    # ------------------------------------------------------------------

    def get_weighted_avg_tdm(
        self,
        ispin: int,
        iband_i: int,
        iband_j: int,
        kweights: Union[np.ndarray, Sequence[float]],
    ) -> dict:
        """k-weight-averaged TDM over irreducible k-points.

        Parameters
        ----------
        ispin : int
        iband_i : int
        iband_j : int
        kweights : array-like, shape (nkpts,)
            Raw weights from IBZKPT.  Normalised internally.

        Returns
        -------
        dict with keys:
            ``"avg_tdm_components"`` shape (3,),
            ``"avg_tdm_magnitude"`` float,
            ``"avg_E_i"`` / ``"avg_E_j"`` / ``"avg_dE"`` float,
            ``"per_kpoint"`` — full per-k result from :meth:`get_tdm_all_kpoints`.

        Examples
        --------
        >>> kw  = read_ibzkpt_weights("IBZKPT")
        >>> avg = wfc.get_weighted_avg_tdm(1, 638, 639, kw)
        """
        kweights = np.asarray(kweights, dtype=float)
        if kweights.shape[0] != self.nkpts:
            raise ValueError(
                f"kweights length {kweights.shape[0]} != nkpts {self.nkpts}."
            )
        per_kpt = self.get_tdm_all_kpoints(ispin, iband_i, iband_j)
        w = kweights / kweights.sum()
        avg_comp = get_weighted_avg_tdm(per_kpt["tdm_components"], kweights)
        return {
            "avg_tdm_components": avg_comp,
            "avg_tdm_magnitude": float(np.linalg.norm(avg_comp)),
            "avg_E_i": float(np.dot(w, per_kpt["E_i"])),
            "avg_E_j": float(np.dot(w, per_kpt["E_j"])),
            "avg_dE": float(np.dot(w, per_kpt["dE"])),
            "per_kpoint": per_kpt,
        }

    # ------------------------------------------------------------------
    # Cross-state TDM (two different WAVECARs)
    # ------------------------------------------------------------------

    def get_tdm_cross_state(
        self,
        other: "WavecarReader",
        ispin: int,
        ikpt: int,
        iband_i: int,
        iband_j: int,
    ) -> Tuple[float, float, float, np.ndarray]:
        """TDM using wavefunctions from two different WAVECARs (ΔSCF approach).

        Parameters
        ----------
        other : WavecarReader
            Reader for the *final* (excited) state.
        ispin : int
        ikpt  : int
        iband_i : int — band in *this* WAVECAR (initial state)
        iband_j : int — band in *other* WAVECAR (final state)

        Returns
        -------
        E_i, E_j, dE : float
        dipole : np.ndarray, shape (3,), complex128 — Debye
        """
        self._check_index(ispin, ikpt, iband_i)
        other._check_index(ispin, ikpt, iband_j)

        Ei = self.bands[ispin - 1, ikpt - 1, iband_i - 1]
        Ej = other.bands[ispin - 1, ikpt - 1, iband_j - 1]
        dE = Ej - Ei

        if np.isclose(dE, 0.0):
            return Ei, Ej, dE, np.zeros(3, dtype=complex)

        k0 = self.kvecs[ikpt - 1]
        G0 = self.gvectors(ikpt)
        Gk = np.dot(G0 + k0, self.Bcell * TPI)

        CG_i = self.read_band_coeff(ispin, ikpt, iband_i)
        CG_j = other.read_band_coeff(ispin, ikpt, iband_j)
        nmin = min(len(CG_i), len(CG_j))
        ovlap = CG_j[:nmin].conj() * CG_i[:nmin]

        if self._lgam:
            mom = np.sum(ovlap[:, None] * Gk[:nmin], axis=0)
            mom -= np.sum(ovlap[:, None].conj() * Gk[:nmin], axis=0)
            mom /= 2.0
        else:
            mom = np.sum(ovlap[:, None] * Gk[:nmin], axis=0)

        dipole = -1j / (dE / (2.0 * RYTOEV)) * mom * AUTOA * AUTDEBYE
        return Ei, Ej, dE, dipole

    def get_tdm_cross_state_all_kpoints(
        self,
        other: "WavecarReader",
        ispin: int,
        iband_i: int,
        iband_j: int,
    ) -> dict:
        """Cross-state TDM at every k-point.

        Returns the same dict structure as :meth:`get_tdm_all_kpoints`.
        """
        nkpts = min(self.nkpts, other.nkpts)
        Ei_arr = np.zeros(nkpts)
        Ej_arr = np.zeros(nkpts)
        dE_arr = np.zeros(nkpts)
        tdm_comp = np.zeros((nkpts, 3))

        for ikpt in range(1, nkpts + 1):
            Ei, Ej, dE, dip = self.get_tdm_cross_state(
                other, ispin, ikpt, iband_i, iband_j
            )
            Ei_arr[ikpt - 1] = Ei
            Ej_arr[ikpt - 1] = Ej
            dE_arr[ikpt - 1] = dE
            tdm_comp[ikpt - 1] = np.abs(dip)

        return {
            "ispin": ispin,
            "iband_i": iband_i,
            "iband_j": iband_j,
            "kvecs": self.kvecs[:nkpts].copy(),
            "E_i": Ei_arr,
            "E_j": Ej_arr,
            "dE": dE_arr,
            "tdm_components": tdm_comp,
            "tdm_magnitude": np.linalg.norm(tdm_comp, axis=1),
        }

    # ------------------------------------------------------------------
    # All occupied→unoccupied transitions
    # ------------------------------------------------------------------

    def get_all_transitions(
        self,
        ispin: int,
        kweights: Optional[Union[np.ndarray, Sequence[float]]] = None,
        mode: str = "occupation",
        occ_bands: Optional[Tuple[int, int]] = None,
        unocc_bands: Optional[Tuple[int, int]] = None,
        occ_band_list: Optional[Sequence[int]] = None,
        unocc_band_list: Optional[Sequence[int]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        fermi_level: float = 0.0,
        occ_threshold: float = 0.5,
        min_tdm: float = 0.0,
        pool_bands: Optional[Sequence[int]] = None,
    ) -> dict:
        """Calculate TDMs for all occupied→unoccupied band pairs.

        Parameters
        ----------
        ispin : int
        kweights : array-like, optional
            Raw k-point weights.  Equal weights used if ``None``.
        mode : {'occupation', 'energy', 'band_range', 'band_list'}
        occ_bands : (lo, hi) inclusive 1-based range for initial states.
            Required when ``mode='band_range'``.
        unocc_bands : (lo, hi) inclusive range for final states.
        occ_band_list, unocc_band_list : sequence of int.
            Required when ``mode='band_list'``.
        energy_range : (E_min, E_max) in eV relative to ``fermi_level``.
            Required when ``mode='energy'``.
        fermi_level : float — eV (absolute).
        occ_threshold : float — minimum occupancy to classify as occupied.
        min_tdm : float — discard transitions with ``|TDM| < min_tdm`` (Debye).
        pool_bands : sequence of int, optional
            Pre-filter: only bands in this list are ever considered.

        Returns
        -------
        dict with keys:
            ``"metadata"``,
            ``"per_kpoint"`` — list of dicts with transitions per k-point,
            ``"pair_summary"`` — BZ-averaged per (i, j) pair,
            ``"strongest_transitions"`` — sorted by descending |TDM|.

        Examples
        --------
        >>> kw = read_ibzkpt_weights("IBZKPT")
        >>> result = wfc.get_all_transitions(1, kw, mode='occupation')
        >>> result = wfc.get_all_transitions(1, kw, mode='band_range',
        ...     occ_bands=(630, 638), unocc_bands=(639, 645))
        """
        if kweights is None:
            kweights = np.ones(self.nkpts)
        kweights = np.asarray(kweights, dtype=float)
        if kweights.shape[0] != self.nkpts:
            raise ValueError(
                f"kweights length {kweights.shape[0]} != nkpts {self.nkpts}."
            )
        w_norm = kweights / kweights.sum()

        if mode == "band_range" and (occ_bands is None or unocc_bands is None):
            raise ValueError("mode='band_range' requires occ_bands and unocc_bands.")
        if mode == "band_list" and (occ_band_list is None or unocc_band_list is None):
            raise ValueError(
                "mode='band_list' requires occ_band_list and unocc_band_list."
            )
        if mode == "energy" and energy_range is None:
            raise ValueError("mode='energy' requires energy_range=(E_min, E_max).")

        pool_set = set(int(b) for b in pool_bands) if pool_bands is not None else None
        per_kpoint_results = []
        pair_acc: dict = {}

        for ikpt in range(1, self.nkpts + 1):
            occ_list, unocc_list = _select_band_pairs_at_kpoint(
                self.bands,
                self.occs,
                ispin,
                ikpt,
                mode=mode,
                occ_bands=occ_bands,
                unocc_bands=unocc_bands,
                occ_band_list=occ_band_list,
                unocc_band_list=unocc_band_list,
                energy_range=energy_range,
                fermi_level=fermi_level,
                occ_threshold=occ_threshold,
            )
            if pool_set is not None:
                occ_list = [b for b in occ_list if b in pool_set]
                unocc_list = [b for b in unocc_list if b in pool_set]

            w_k = float(w_norm[ikpt - 1])
            transitions = []

            for bi in occ_list:
                for bj in unocc_list:
                    if bi == bj:
                        continue
                    Ei, Ej, dE, dip = self.get_dipole_matrix(ispin, ikpt, bi, bj)
                    tdm_comp = np.abs(dip)
                    tdm_mag = float(np.linalg.norm(tdm_comp))
                    if tdm_mag < min_tdm:
                        continue
                    transitions.append(
                        {
                            "iband_i": bi,
                            "iband_j": bj,
                            "E_i": float(Ei),
                            "E_j": float(Ej),
                            "dE": float(dE),
                            "tdm_components": tdm_comp.tolist(),
                            "tdm_magnitude": tdm_mag,
                        }
                    )
                    key = (bi, bj)
                    if key not in pair_acc:
                        pair_acc[key] = {
                            "sum_comp": np.zeros(3),
                            "sum_Ei": 0.0,
                            "sum_Ej": 0.0,
                            "sum_dE": 0.0,
                            "sum_w": 0.0,
                            "n": 0,
                        }
                    pair_acc[key]["sum_comp"] += tdm_comp * w_k
                    pair_acc[key]["sum_Ei"] += float(Ei) * w_k
                    pair_acc[key]["sum_Ej"] += float(Ej) * w_k
                    pair_acc[key]["sum_dE"] += float(dE) * w_k
                    pair_acc[key]["sum_w"] += w_k
                    pair_acc[key]["n"] += 1

            per_kpoint_results.append(
                {
                    "ikpt": ikpt,
                    "kvec": self.kvecs[ikpt - 1].tolist(),
                    "weight": w_k,
                    "n_occ_bands": len(occ_list),
                    "n_unocc_bands": len(unocc_list),
                    "transitions": transitions,
                }
            )

        pair_summary = []
        for (bi, bj), acc in sorted(pair_acc.items()):
            sw = acc["sum_w"]
            if sw == 0:
                continue
            avg_comp = acc["sum_comp"] / sw
            pair_summary.append(
                {
                    "iband_i": bi,
                    "iband_j": bj,
                    "n_kpoints": acc["n"],
                    "avg_tdm_components": avg_comp.tolist(),
                    "avg_tdm_magnitude": float(np.linalg.norm(avg_comp)),
                    "avg_E_i": acc["sum_Ei"] / sw,
                    "avg_E_j": acc["sum_Ej"] / sw,
                    "avg_dE": acc["sum_dE"] / sw,
                }
            )

        strongest = sorted(
            pair_summary, key=lambda x: x["avg_tdm_magnitude"], reverse=True
        )

        return {
            "metadata": {
                "source": str(self._fname),
                "ispin": ispin,
                "nkpts": self.nkpts,
                "mode": mode,
                "occ_bands": occ_bands,
                "unocc_bands": unocc_bands,
                "occ_band_list": list(occ_band_list) if occ_band_list else None,
                "unocc_band_list": list(unocc_band_list) if unocc_band_list else None,
                "energy_range": list(energy_range) if energy_range else None,
                "fermi_level": fermi_level,
                "occ_threshold": occ_threshold,
                "min_tdm": min_tdm,
                "n_unique_pairs": len(pair_summary),
                "kweights": kweights.tolist(),
            },
            "per_kpoint": per_kpoint_results,
            "pair_summary": pair_summary,
            "strongest_transitions": strongest,
        }

    # ------------------------------------------------------------------
    # WAVECAR trimming / compact saving
    # ------------------------------------------------------------------

    def trim_save_wavecar(
        self,
        bands: Sequence[int],
        outfile: Union[str, Path] = "WAVECAR_trim",
        compact: bool = False,
    ) -> List[int]:
        """Write a WAVECAR that keeps all band indices but zeros non-selected bands.

        Parameters
        ----------
        bands : sequence of int — 1-based band indices to keep (original numbering).
        outfile : str or Path
        compact : bool — if True, delegate to :meth:`save_compact_wavecar`.

        Returns
        -------
        list of int
            Sorted list of band indices stored with data.

        Examples
        --------
        >>> kept = wfc.trim_save_wavecar([635, 636, 637, 638, 639])
        >>> wfc2 = WavecarReader("WAVECAR_trim")
        >>> res  = wfc2.get_tdm_all_kpoints(1, 638, 639)  # original indices work
        """
        if compact:
            return self.save_compact_wavecar(bands, outfile=outfile)

        bands_set = set(int(b) for b in bands)
        for b in bands_set:
            if not (1 <= b <= self.nbands):
                raise IndexError(f"Band {b} out of range [1, {self.nbands}].")
        bands_sorted = sorted(bands_set)
        outfile = Path(outfile)
        zero_cg = np.zeros(1, dtype=self._WFPrec)

        with open(outfile, "wb") as fout:
            rec0 = np.array([self._recl, self.nspin, self._rtag], dtype=np.float64)
            _write_record(fout, rec0, self._recl)
            rec1 = np.concatenate(
                [
                    [float(self.nkpts), float(self.nbands), self.encut],
                    self.Acell.flatten(),
                ]
            )
            _write_record(fout, rec1, self._recl)

            for ispin in range(1, self.nspin + 1):
                for ikpt in range(1, self.nkpts + 1):
                    nplw = self._nplws[ikpt - 1]
                    kvec = self.kvecs[ikpt - 1]
                    kpt_header = np.empty(4 + 3 * self.nbands, dtype=np.float64)
                    kpt_header[0] = nplw
                    kpt_header[1:4] = kvec
                    for b in range(1, self.nbands + 1):
                        ei = self.bands[ispin - 1, ikpt - 1, b - 1]
                        oi = self.occs[ispin - 1, ikpt - 1, b - 1]
                        kpt_header[4 + 3 * (b - 1)] = ei
                        kpt_header[4 + 3 * (b - 1) + 1] = 0.0
                        kpt_header[4 + 3 * (b - 1) + 2] = oi
                    _write_record(fout, kpt_header, self._recl)

                    for b in range(1, self.nbands + 1):
                        if b in bands_set:
                            cg = self.read_band_coeff(ispin, ikpt, b).astype(
                                self._WFPrec
                            )
                        else:
                            cg = zero_cg
                        _write_record(fout, cg, self._recl)

        print(f"Trimmed WAVECAR written to: {outfile}")
        print(
            f"  nbands preserved: {self.nbands}  (coefficients for {len(bands_sorted)} bands)"
        )
        print(f"  Bands with data: {bands_sorted}")
        return bands_sorted

    def save_compact_wavecar(
        self,
        bands: Sequence[int],
        outfile: Union[str, Path] = "WAVECAR_compact",
    ) -> List[int]:
        """Write a compact WAVECAR containing ONLY the selected band records.

        Original band indices and ``nbands`` are embedded in the file header
        so no companion file is needed.  :class:`WavecarReader` auto-detects
        this format.

        Parameters
        ----------
        bands : sequence of int — 1-based band indices to store.
        outfile : str or Path

        Returns
        -------
        list of int
            Sorted list of band indices stored.

        Examples
        --------
        >>> wfc.save_compact_wavecar([430, 431], outfile="WAVECAR_gs_compact")
        >>> cwfc = WavecarReader("WAVECAR_gs_compact")
        >>> cwfc.nbands          # original nbands, recovered from header
        """
        bands_set = set(int(b) for b in bands)
        for b in bands_set:
            if not (1 <= b <= self.nbands):
                raise IndexError(f"Band {b} out of range [1, {self.nbands}].")
        bands_sorted = sorted(bands_set)
        nbands_compact = len(bands_sorted)
        outfile = Path(outfile)

        with open(outfile, "wb") as fout:
            rec0 = np.array([self._recl, self.nspin, self._rtag], dtype=np.float64)
            _write_record(fout, rec0, self._recl)
            rec1 = np.concatenate(
                [
                    [float(self.nkpts), float(nbands_compact), self.encut],
                    self.Acell.flatten(),
                ]
            )
            _write_record(fout, rec1, self._recl)

            for ispin in range(1, self.nspin + 1):
                for ikpt in range(1, self.nkpts + 1):
                    nplw = self._nplws[ikpt - 1]
                    kvec = self.kvecs[ikpt - 1]
                    kpt_header = np.zeros(4 + 3 * nbands_compact + 2, dtype=np.float64)
                    kpt_header[0] = nplw
                    kpt_header[1:4] = kvec
                    for ci, b in enumerate(bands_sorted):
                        ei = self.bands[ispin - 1, ikpt - 1, b - 1]
                        oi = self.occs[ispin - 1, ikpt - 1, b - 1]
                        kpt_header[4 + 3 * ci] = ei
                        kpt_header[4 + 3 * ci + 1] = float(b)
                        kpt_header[4 + 3 * ci + 2] = oi
                    kpt_header[-2] = self._COMPACT_MARKER
                    kpt_header[-1] = float(self.nbands)
                    _write_record(fout, kpt_header, self._recl)

                    for b in bands_sorted:
                        cg = self.read_band_coeff(ispin, ikpt, b).astype(self._WFPrec)
                        _write_record(fout, cg, self._recl)

        mb = outfile.stat().st_size / 1e6
        print(f"Compact WAVECAR written to: {outfile}  ({mb:.2f} MB)")
        print(f"  Stored {nbands_compact} of {self.nbands} bands: {bands_sorted}")
        return bands_sorted

    # ------------------------------------------------------------------
    # JSON / HDF5 export
    # ------------------------------------------------------------------

    def save_to_json(
        self,
        bands: Optional[Sequence[int]] = None,
        outfile: Union[str, Path] = "wavecar_info.json",
        save_coeffs: bool = False,
    ) -> None:
        """Export WAVECAR metadata (and optionally coefficients) to JSON.

        Parameters
        ----------
        bands : sequence of int, optional — 1-based indices.  All if ``None``.
        outfile : str or Path
        save_coeffs : bool — include plane-wave coefficients.

        Examples
        --------
        >>> wfc.save_to_json(bands=[638, 639], outfile="bands.json")
        """
        if bands is None:
            bands = list(range(1, self.nbands + 1))
        bands = sorted(set(int(b) for b in bands))

        data: dict = {
            "source": str(self._fname),
            "nspin": self.nspin,
            "nkpts": self.nkpts,
            "nbands_original": self.nbands,
            "encut": self.encut,
            "Acell": self.Acell.tolist(),
            "Bcell": self.Bcell.tolist(),
            "Omega": float(self.Omega),
            "kvecs": self.kvecs.tolist(),
            "bands_selected": bands,
            "eigenvalues": {},
            "occupancies": {},
        }
        for ispin in range(1, self.nspin + 1):
            sk = f"spin{ispin}"
            data["eigenvalues"][sk] = {}
            data["occupancies"][sk] = {}
            for ikpt in range(1, self.nkpts + 1):
                kk = f"kpt{ikpt}"
                data["eigenvalues"][sk][kk] = [
                    float(self.bands[ispin - 1, ikpt - 1, b - 1]) for b in bands
                ]
                data["occupancies"][sk][kk] = [
                    float(self.occs[ispin - 1, ikpt - 1, b - 1]) for b in bands
                ]

        if save_coeffs:
            data["coefficients"] = {}
            for ispin in range(1, self.nspin + 1):
                sk = f"spin{ispin}"
                data["coefficients"][sk] = {}
                for ikpt in range(1, self.nkpts + 1):
                    kk = f"kpt{ikpt}"
                    data["coefficients"][sk][kk] = {}
                    for b in bands:
                        cg = self.read_band_coeff(ispin, ikpt, b)
                        data["coefficients"][sk][kk][f"band{b}"] = [
                            [float(c.real), float(c.imag)] for c in cg
                        ]

        with open(outfile, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"JSON export written to: {outfile}")

    def save_to_h5(
        self,
        bands: Optional[Sequence[int]] = None,
        outfile: Union[str, Path] = "wavecar_trim.h5",
    ) -> None:
        """Export WAVECAR data to HDF5.  Requires h5py.

        Parameters
        ----------
        bands : sequence of int, optional
        outfile : str or Path

        Examples
        --------
        >>> wfc.save_to_h5(bands=[635, 636, 637, 638, 639], outfile="trim.h5")
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required. Install with: pip install h5py")

        if bands is None:
            bands = list(range(1, self.nbands + 1))
        bands = sorted(set(int(b) for b in bands))

        with h5py.File(outfile, "w") as hf:
            hf.attrs["source"] = str(self._fname)
            hf.attrs["nspin"] = self.nspin
            hf.attrs["nkpts"] = self.nkpts
            hf.attrs["nbands_original"] = self.nbands
            hf.attrs["encut"] = self.encut
            hf.attrs["ngrid"] = np.array(self._ngrid, dtype=int)
            hf.create_dataset("Acell", data=self.Acell)
            hf.create_dataset("Bcell", data=self.Bcell)
            hf.create_dataset("kvecs", data=self.kvecs)
            hf.create_dataset("bands_selected", data=np.array(bands, dtype=int))
            hf.create_dataset("eigenvalues", data=self.bands)
            hf.create_dataset("occupancies", data=self.occs)

            grp = hf.create_group("coefficients")
            for ispin in range(1, self.nspin + 1):
                sg = grp.create_group(f"spin{ispin}")
                for ikpt in range(1, self.nkpts + 1):
                    kg = sg.create_group(f"kpt{ikpt}")
                    for b in bands:
                        cg = self.read_band_coeff(ispin, ikpt, b)
                        kg.create_dataset(f"band{b}", data=cg)

        print(f"HDF5 export written to: {outfile}  ({len(bands)} bands)")

    # ------------------------------------------------------------------
    # Real-space wavefunction
    # ------------------------------------------------------------------

    def wfc_r(
        self,
        ispin: int,
        ikpt: int,
        iband: int,
        ngrid=None,
    ):
        """Real-space pseudo-wavefunction via inverse FFT.

        Parameters
        ----------
        ispin, ikpt, iband : int — 1-based indices.
        ngrid : array-like of int, optional
            FFT grid (Nx, Ny, Nz).  Defaults to 2 × minimum grid.

        Returns
        -------
        phi : np.ndarray, shape (Nx, Ny, Nz), dtype complex128
            For SOC, a list of two spinor arrays.

        Examples
        --------
        >>> phi     = wfc.wfc_r(1, 1, 638)
        >>> density = np.abs(phi) ** 2
        """
        self._check_index(ispin, ikpt, iband)
        if ngrid is None:
            ngrid = self._ngrid * 2
        ngrid = np.asarray(ngrid, dtype=int)
        G0 = self.gvectors(ikpt)

        if self._lsoc:
            nplw = self._nplws[ikpt - 1]
            cg_full = self.read_band_coeff(ispin, ikpt, iband)
            half = nplw // 2
            result = []
            for cg_part in (cg_full[:half], cg_full[half:]):
                wfc_g = np.zeros(ngrid, dtype=np.complex128)
                for G, c in zip(G0, cg_part):
                    idx = tuple(int(g) % n for g, n in zip(G, ngrid))
                    wfc_g[idx] += c
                result.append(np.fft.ifftn(wfc_g) * np.prod(ngrid))
            return result

        cg = self.read_band_coeff(ispin, ikpt, iband)
        if self._lgam:
            wfc_g = np.zeros(ngrid, dtype=np.complex128)
            for G, c in zip(G0, cg):
                idx = tuple(int(g) % n for g, n in zip(G, ngrid))
                wfc_g[idx] = c
            for G, c in zip(G0, cg):
                idx = tuple((-int(g)) % n for g, n in zip(G, ngrid))
                wfc_g[idx] = c.conj()
            return np.fft.ifftn(wfc_g).real * np.prod(ngrid)
        else:
            wfc_g = np.zeros(ngrid, dtype=np.complex128)
            for G, c in zip(G0, cg):
                idx = tuple(int(g) % n for g, n in zip(G, ngrid))
                wfc_g[idx] = c
            return np.fft.ifftn(wfc_g) * np.prod(ngrid)


# ===========================================================================
# VaspwaveH5Reader — VASP 6 HDF5 WAVECAR
# ===========================================================================


class VaspwaveH5Reader:
    """Read VASP wavefunctions from a vaspwave.h5 HDF5 file (VASP 6+).

    Expected dataset layout (written by :meth:`WavecarReader.save_to_h5`)::

        /Acell, /Bcell, /kvecs
        /eigenvalues   (nspin, nkpts, nbands) — full array, original indices
        /occupancies   (nspin, nkpts, nbands)
        /bands_selected  (m,) int
        /coefficients/spin{i}/kpt{j}/band{b}   — original band index in path

    Parameters
    ----------
    h5file : str or Path

    Examples
    --------
    >>> reader = VaspwaveH5Reader("wavecar_trim.h5")
    >>> res = reader.get_tdm_all_kpoints(1, 638, 639)
    """

    def __init__(self, h5file: Union[str, Path]) -> None:
        import importlib.util

        if importlib.util.find_spec("h5py") is None:
            raise ImportError("h5py is required. Install with: pip install h5py")
        self._fname = Path(h5file)
        self._load_h5()

    def _load_h5(self) -> None:
        import h5py

        with h5py.File(self._fname, "r") as hf:
            self.Acell = hf["Acell"][:]
            self.Bcell = hf["Bcell"][:]
            self.kvecs = hf["kvecs"][:]
            self.bands = hf["eigenvalues"][:]
            self.occs = hf["occupancies"][:]
            self.encut = float(
                hf.attrs.get("encut", hf["encut"][()] if "encut" in hf else 520.0)
            )
            self._bands_stored = set(int(b) for b in hf["bands_selected"][:])
            self._ngrid = (
                tuple(int(x) for x in hf.attrs["ngrid"])
                if "ngrid" in hf.attrs
                else None
            )
        self.nspin, self.nkpts, self.nbands = self.bands.shape
        self.Omega = np.abs(np.linalg.det(self.Acell))

    def _read_coeff(self, ispin: int, ikpt: int, iband: int) -> np.ndarray:
        import h5py

        key = f"coefficients/spin{ispin}/kpt{ikpt}/band{iband}"
        with h5py.File(self._fname, "r") as hf:
            if key not in hf:
                raise KeyError(
                    f"Band {iband} coefficients not in {self._fname}. "
                    f"Stored: {sorted(self._bands_stored)}"
                )
            return hf[key][:]

    def _check_band_available(self, iband: int) -> None:
        if iband not in self._bands_stored:
            raise KeyError(
                f"Band {iband} not stored. Available: {sorted(self._bands_stored)}"
            )

    def gvectors(self, ikpt: int = 1) -> np.ndarray:
        """Reconstruct G-vectors in VASP ordering from stored metadata."""
        kvec = self.kvecs[ikpt - 1]
        if self._ngrid is not None:
            ngrid = np.array(self._ngrid, dtype=int)
        else:
            Anorm = np.linalg.norm(self.Acell, axis=1)
            cutof = np.ceil(sqrt(self.encut / RYTOEV) / (TPI / (Anorm / AUTOA)))
            ngrid = np.array(2 * cutof + 1, dtype=int)

        fx = np.arange(ngrid[0], dtype=int)
        fy = np.arange(ngrid[1], dtype=int)
        fz = np.arange(ngrid[2], dtype=int)
        fx[ngrid[0] // 2 + 1 :] -= ngrid[0]
        fy[ngrid[1] // 2 + 1 :] -= ngrid[1]
        fz[ngrid[2] // 2 + 1 :] -= ngrid[2]

        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape(3, -1)
        kgrid = np.column_stack([gx, gy, gz]).astype(float)
        ke = (
            HSQDTM
            * np.linalg.norm(
                np.dot(kgrid + kvec[np.newaxis, :], TPI * self.Bcell), axis=1
            )
            ** 2
        )
        return np.asarray(kgrid[ke < self.encut], dtype=int)

    def get_momentum_matrix(
        self, ispin: int, ikpt: int, iband_i: int, iband_j: int
    ) -> np.ndarray:
        """Momentum matrix element from h5 coefficients."""
        self._check_band_available(iband_i)
        self._check_band_available(iband_j)
        k0 = self.kvecs[ikpt - 1]
        G0 = self.gvectors(ikpt)
        Gk = np.dot(G0 + k0, self.Bcell * TPI)
        CG_i = self._read_coeff(ispin, ikpt, iband_i)
        CG_j = self._read_coeff(ispin, ikpt, iband_j)
        nmin = min(len(CG_i), len(CG_j), len(Gk))
        ovlap = CG_j[:nmin].conj() * CG_i[:nmin]
        return np.sum(ovlap[:, None] * Gk[:nmin], axis=0)

    def get_tdm_cross_state(
        self,
        other: "VaspwaveH5Reader",
        ispin: int,
        ikpt: int,
        iband_i: int,
        iband_j: int,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Cross-state TDM between two H5 readers."""
        self._check_band_available(iband_i)
        other._check_band_available(iband_j)

        Ei = float(self.bands[ispin - 1, ikpt - 1, iband_i - 1])
        Ej = float(other.bands[ispin - 1, ikpt - 1, iband_j - 1])
        dE = Ej - Ei
        if np.isclose(dE, 0.0):
            return Ei, Ej, dE, np.zeros(3, dtype=complex)

        k0 = self.kvecs[ikpt - 1]
        G0 = self.gvectors(ikpt)
        Gk = np.dot(G0 + k0, self.Bcell * TPI)
        CG_i = self._read_coeff(ispin, ikpt, iband_i)
        CG_j = other._read_coeff(ispin, ikpt, iband_j)
        nmin = min(len(CG_i), len(CG_j), len(Gk))
        mom = np.einsum("x,xv->v", CG_i[:nmin].conj() * CG_j[:nmin], Gk[:nmin])

        dipole = -1j / (dE / RYTOEV / 2.0) * mom * AUTOA * AUTDEBYE
        return Ei, Ej, dE, dipole

    def get_tdm_all_kpoints(self, ispin: int, iband_i: int, iband_j: int) -> dict:
        """TDM at every k-point (same interface as WavecarReader)."""
        self._check_band_available(iband_i)
        self._check_band_available(iband_j)

        Ei_arr = np.zeros(self.nkpts)
        Ej_arr = np.zeros(self.nkpts)
        dE_arr = np.zeros(self.nkpts)
        tdm_comp = np.zeros((self.nkpts, 3))

        for ikpt in range(1, self.nkpts + 1):
            Ei = float(self.bands[ispin - 1, ikpt - 1, iband_i - 1])
            Ej = float(self.bands[ispin - 1, ikpt - 1, iband_j - 1])
            dE = Ej - Ei
            if np.isclose(dE, 0.0):
                Ei_arr[ikpt - 1] = Ei
                Ej_arr[ikpt - 1] = Ej
                continue
            k0 = self.kvecs[ikpt - 1]
            G0 = self.gvectors(ikpt)
            Gk = np.dot(G0 + k0, self.Bcell * TPI)
            CG_i = self._read_coeff(ispin, ikpt, iband_i)
            CG_j = self._read_coeff(ispin, ikpt, iband_j)
            nmin = min(len(CG_i), len(CG_j), len(Gk))
            mom = np.einsum("x,xv->v", CG_j[:nmin].conj() * CG_i[:nmin], Gk[:nmin])
            dip = -1j / (dE / RYTOEV / 2.0) * mom * AUTOA * AUTDEBYE
            Ei_arr[ikpt - 1] = Ei
            Ej_arr[ikpt - 1] = Ej
            dE_arr[ikpt - 1] = dE
            tdm_comp[ikpt - 1] = np.abs(dip)

        return {
            "ispin": ispin,
            "iband_i": iband_i,
            "iband_j": iband_j,
            "kvecs": self.kvecs.copy(),
            "E_i": Ei_arr,
            "E_j": Ej_arr,
            "dE": dE_arr,
            "tdm_components": tdm_comp,
            "tdm_magnitude": np.linalg.norm(tdm_comp, axis=1),
        }


# ===========================================================================
# Band selection helper
# ===========================================================================


def select_bands(
    wfc,
    ispin: int = 1,
    mode: str = "near_fermi",
    n_occ: int = 10,
    n_unocc: int = 10,
    below_homo_ev: float = 2.0,
    above_lumo_ev: float = 2.0,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
    band_range: Optional[Tuple[int, int]] = None,
    band_list: Optional[Sequence[int]] = None,
    fermi_level: Optional[float] = None,
    occ_threshold: float = 0.5,
) -> List[int]:
    """Select band indices using one of several strategies.

    Modes
    -----
    ``'all'``
        Every band.
    ``'band_list'``
        Explicit 1-based list (``band_list``).
    ``'band_range'``
        Contiguous range ``band_range=(lo, hi)`` inclusive.
    ``'occupation'``
        All bands sorted by occupancy character.
    ``'near_fermi'``
        ``n_occ`` bands closest to HOMO + ``n_unocc`` closest to LUMO.
    ``'homo_lumo_range'``
        All bands within ``[HOMO − below_homo_ev, LUMO + above_lumo_ev]``.
    ``'energy_window'``
        All bands within ``[energy_min, energy_max]`` (absolute eV).

    Parameters
    ----------
    wfc : WavecarReader or VaspwaveH5Reader
    ispin : int
    mode : str
    n_occ, n_unocc : int — band counts for ``'near_fermi'``.
    below_homo_ev, above_lumo_ev : float — half-widths for ``'homo_lumo_range'``.
    energy_min, energy_max : float — absolute bounds for ``'energy_window'``.
    band_range : (int, int) — for ``'band_range'`` mode.
    band_list : sequence of int — for ``'band_list'`` mode.
    fermi_level : float, optional — if given, overrides occupancy-based detection.
    occ_threshold : float — occupancy cutoff.

    Returns
    -------
    list of int
        Sorted 1-based band indices.

    Examples
    --------
    >>> bands = select_bands(wfc, 1, mode='near_fermi', n_occ=10, n_unocc=10)
    >>> bands = select_bands(wfc, 1, mode='homo_lumo_range', below_homo_ev=2.0)
    >>> bands = select_bands(wfc, 1, mode='band_range', band_range=(630, 660))
    """
    s = ispin - 1
    nbands = int(wfc.bands.shape[2])

    if mode == "all":
        return list(range(1, nbands + 1))

    if mode == "band_list":
        if band_list is None:
            raise ValueError("band_list is required for mode='band_list'.")
        return sorted(int(b) for b in band_list)

    if mode == "band_range":
        if band_range is None:
            raise ValueError("band_range=(lo, hi) is required for mode='band_range'.")
        lo, hi = int(band_range[0]), int(band_range[1])
        if not (1 <= lo <= hi <= nbands):
            raise ValueError(f"band_range ({lo}, {hi}) out of [1, {nbands}].")
        return list(range(lo, hi + 1))

    if mode == "occupation":
        avg_occ = wfc.occs[s].mean(axis=0)
        occ = sorted(np.where(avg_occ >= occ_threshold)[0])
        unocc = sorted(np.where(avg_occ < occ_threshold)[0])
        return [i + 1 for i in occ + unocc]

    mean_e = wfc.bands[s].mean(axis=0)

    if fermi_level is not None:
        occ_mask = mean_e <= fermi_level
    else:
        occ_mask = wfc.occs[s].mean(axis=0) >= occ_threshold

    occ_idx = np.where(occ_mask)[0]
    unocc_idx = np.where(~occ_mask)[0]

    if len(occ_idx) == 0:
        raise ValueError(
            "No occupied bands found. Provide fermi_level or adjust occ_threshold."
        )
    if len(unocc_idx) == 0:
        raise ValueError("No unoccupied bands found.")

    homo_e = mean_e[occ_idx].max()
    lumo_e = mean_e[unocc_idx].min()

    if mode == "near_fermi":
        occ_sorted = sorted(occ_idx, key=lambda i: -mean_e[i])
        unocc_sorted = sorted(unocc_idx, key=lambda i: mean_e[i])
        sel = sorted(set(occ_sorted[:n_occ]) | set(unocc_sorted[:n_unocc]))
        return [i + 1 for i in sel]

    if mode == "homo_lumo_range":
        emin = homo_e - below_homo_ev
        emax = lumo_e + above_lumo_ev
        return [i + 1 for i in np.where((mean_e >= emin) & (mean_e <= emax))[0]]

    if mode == "energy_window":
        if energy_min is None or energy_max is None:
            raise ValueError(
                "energy_min and energy_max required for mode='energy_window'."
            )
        return [
            i + 1 for i in np.where((mean_e >= energy_min) & (mean_e <= energy_max))[0]
        ]

    raise ValueError(
        f"Unknown mode '{mode}'. Choose from: "
        "all, band_list, band_range, occupation, "
        "near_fermi, homo_lumo_range, energy_window."
    )


# ===========================================================================
# IPR / PR calculations
# ===========================================================================


def _ipr_from_density(phi_abs2: np.ndarray) -> Tuple[float, float, float]:
    """Compute IPR, PR, and normalised PR from |φ|² array."""
    n_grid = phi_abs2.size
    num = float(np.sum(phi_abs2**2))
    den = float(np.sum(phi_abs2)) ** 2
    if den == 0:
        return float("inf"), 0.0, 0.0
    ipr = num / den
    pr = 1.0 / ipr if ipr > 0 else float("inf")
    pr_norm = pr / n_grid if n_grid > 0 else 0.0
    return ipr, pr, pr_norm


def compute_ipr_band(
    wfc,
    ispin: int,
    ikpt: int,
    iband: int,
    ngrid=None,
) -> Tuple[float, float, float]:
    """IPR, PR, and normalised PR for a single KS state.

    .. math::

        \\text{IPR} = \\frac{\\sum_n |\\phi(n)|^4}{(\\sum_n |\\phi(n)|^2)^2}

    Parameters
    ----------
    wfc : WavecarReader or compatible
        Must expose ``wfc_r(ispin, ikpt, iband, ngrid)``.
    ispin, ikpt, iband : int — 1-based.
    ngrid : optional — FFT grid size.

    Returns
    -------
    ipr : float
    pr  : float  — participation ratio = 1/IPR
    pr_norm : float  — PR / N_grid ∈ [0, 1]

    Examples
    --------
    >>> ipr, pr, pr_norm = compute_ipr_band(wfc, 1, 1, 638)
    """
    phi = wfc.wfc_r(ispin, ikpt, iband, ngrid=ngrid)
    if isinstance(phi, list):
        phi_abs2 = sum(np.abs(p) ** 2 for p in phi)
    else:
        phi_abs2 = np.abs(phi) ** 2
    return _ipr_from_density(phi_abs2)


def compute_ipr_all(
    wfc,
    ispin: int,
    kweights: Optional[Union[np.ndarray, Sequence[float]]] = None,
    bands: Optional[Sequence[int]] = None,
    select_mode: str = "all",
    n_occ: int = 10,
    n_unocc: int = 10,
    below_homo_ev: float = 2.0,
    above_lumo_ev: float = 2.0,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
    band_range: Optional[Tuple[int, int]] = None,
    band_list: Optional[Sequence[int]] = None,
    fermi_level: Optional[float] = None,
    occ_threshold: float = 0.5,
    energy_range: Optional[Tuple[float, float]] = None,
    ngrid=None,
    verbose: bool = True,
) -> dict:
    """IPR and PR for multiple bands at all k-points.

    Accepts the same ``select_mode`` vocabulary as :func:`select_bands`,
    or an explicit ``bands`` list.

    Parameters
    ----------
    wfc : WavecarReader or compatible
    ispin : int
    kweights : array-like, optional
    bands : sequence of int, optional — explicit list, overrides select_mode.
    select_mode : str — passed to :func:`select_bands` when ``bands`` is None.
    n_occ, n_unocc : int
    below_homo_ev, above_lumo_ev : float
    energy_min, energy_max : float
    band_range : (int, int)
    band_list : sequence of int
    fermi_level : float, optional
    occ_threshold : float
    energy_range : (float, float), optional — legacy shortcut.
    ngrid : optional
    verbose : bool

    Returns
    -------
    dict with keys:
        ``"metadata"``,
        ``"per_band_per_kpoint"`` — list of dicts per (band, kpt),
        ``"band_summary"``        — list of dicts per band (k-weighted avgs),
        ``"kweights"``            — np.ndarray.

    Examples
    --------
    >>> result = compute_ipr_all(wfc, 1, kweights=kw,
    ...     select_mode='near_fermi', n_occ=10, n_unocc=10)
    >>> result = compute_ipr_all(wfc, 1, kweights=kw,
    ...     bands=[635, 636, 637, 638, 639])
    """
    nkpts = wfc.nkpts
    nbands = wfc.nbands

    if kweights is None:
        kweights = np.ones(nkpts)
    kweights = np.asarray(kweights, dtype=float)
    w_norm = kweights / kweights.sum()

    if bands is not None:
        band_list_resolved = sorted(set(int(b) for b in bands))
    else:
        if energy_range is not None and select_mode == "all":
            fl = fermi_level if fermi_level is not None else 0.0
            energy_min = fl + energy_range[0]
            energy_max = fl + energy_range[1]
            select_mode = "energy_window"

        band_list_resolved = select_bands(
            wfc,
            ispin=ispin,
            mode=select_mode,
            n_occ=n_occ,
            n_unocc=n_unocc,
            below_homo_ev=below_homo_ev,
            above_lumo_ev=above_lumo_ev,
            energy_min=energy_min,
            energy_max=energy_max,
            band_range=band_range,
            band_list=band_list,
            fermi_level=fermi_level,
            occ_threshold=occ_threshold,
        )
        if not band_list_resolved:
            raise ValueError("Band selection returned an empty list.")

    per_band_per_kpoint = []
    band_acc: Dict[int, List] = {b: [] for b in band_list_resolved}
    total = len(band_list_resolved) * nkpts
    done = 0

    for iband in band_list_resolved:
        for ikpt in range(1, nkpts + 1):
            ipr, pr, pr_norm = compute_ipr_band(wfc, ispin, ikpt, iband, ngrid=ngrid)
            energy = float(wfc.bands[ispin - 1, ikpt - 1, iband - 1])
            occ = float(wfc.occs[ispin - 1, ikpt - 1, iband - 1])
            w_k = float(w_norm[ikpt - 1])
            per_band_per_kpoint.append(
                {
                    "iband": iband,
                    "ikpt": ikpt,
                    "kvec": wfc.kvecs[ikpt - 1].tolist(),
                    "energy": energy,
                    "occupancy": occ,
                    "ipr": ipr,
                    "pr": pr,
                    "pr_norm": pr_norm,
                }
            )
            band_acc[iband].append((ipr, pr, pr_norm, energy, w_k, occ))
            done += 1
            if verbose and done % max(1, total // 20) == 0:
                print(f"  IPR: {done}/{total} ({100 * done // total}%)", end="\r")

    if verbose:
        print(f"  IPR: {total}/{total} (100%)   ")

    band_summary = []
    for iband in band_list_resolved:
        records = band_acc[iband]
        iprs = np.array([r[0] for r in records])
        prs = np.array([r[1] for r in records])
        pr_norms = np.array([r[2] for r in records])
        energies = np.array([r[3] for r in records])
        ws = np.array([r[4] for r in records])
        w_sum = ws.sum()
        band_summary.append(
            {
                "iband": iband,
                "avg_energy": float(np.dot(ws, energies) / w_sum),
                "avg_ipr": float(np.mean(iprs)),
                "avg_pr": float(np.mean(prs)),
                "avg_pr_norm": float(np.mean(pr_norms)),
                "weighted_avg_ipr": float(np.dot(ws, iprs) / w_sum),
                "weighted_avg_pr": float(np.dot(ws, prs) / w_sum),
                "weighted_avg_pr_norm": float(np.dot(ws, pr_norms) / w_sum),
                "min_ipr": float(iprs.min()),
                "max_ipr": float(iprs.max()),
                "std_ipr": float(iprs.std()),
            }
        )

    return {
        "metadata": {
            "ispin": ispin,
            "nkpts": nkpts,
            "nbands": nbands,
            "bands": band_list_resolved,
            "select_mode": select_mode,
            "fermi_level": fermi_level,
            "kweights": kweights.tolist(),
        },
        "per_band_per_kpoint": per_band_per_kpoint,
        "band_summary": band_summary,
        "kweights": kweights,
    }


def compute_ipr_weighted(
    wfc,
    ispin: int,
    iband: int,
    kweights: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ngrid=None,
) -> dict:
    """k-weight-averaged IPR for a single band.

    Parameters
    ----------
    wfc : WavecarReader or compatible
    ispin : int
    iband : int — 1-based.
    kweights : array-like, optional
    ngrid : optional

    Returns
    -------
    dict with keys:
        ``"ipr_per_kpoint"``, ``"avg_ipr"``, ``"weighted_avg_ipr"``,
        ``"avg_pr"``, ``"weighted_avg_pr"``, ``"avg_pr_norm"``,
        ``"weighted_avg_pr_norm"``.

    Examples
    --------
    >>> res = compute_ipr_weighted(wfc, 1, 638, kweights=kw)
    >>> print(res["weighted_avg_ipr"])
    """
    nkpts = wfc.nkpts
    if kweights is None:
        kweights = np.ones(nkpts)
    kweights = np.asarray(kweights, dtype=float)
    w = kweights / kweights.sum()

    ipr_arr = np.zeros(nkpts)
    pr_arr = np.zeros(nkpts)
    pr_norm_arr = np.zeros(nkpts)
    for ikpt in range(1, nkpts + 1):
        ipr, pr, pr_norm = compute_ipr_band(wfc, ispin, ikpt, iband, ngrid=ngrid)
        ipr_arr[ikpt - 1] = ipr
        pr_arr[ikpt - 1] = pr
        pr_norm_arr[ikpt - 1] = pr_norm

    return {
        "iband": iband,
        "ispin": ispin,
        "ipr_per_kpoint": ipr_arr.tolist(),
        "pr_per_kpoint": pr_arr.tolist(),
        "pr_norm_per_kpoint": pr_norm_arr.tolist(),
        "avg_ipr": float(np.mean(ipr_arr)),
        "weighted_avg_ipr": float(np.dot(w, ipr_arr)),
        "avg_pr": float(np.mean(pr_arr)),
        "weighted_avg_pr": float(np.dot(w, pr_arr)),
        "avg_pr_norm": float(np.mean(pr_norm_arr)),
        "weighted_avg_pr_norm": float(np.dot(w, pr_norm_arr)),
    }


def save_ipr_json(
    result: dict,
    outfile: Union[str, Path] = "ipr_result.json",
) -> None:
    """Save :func:`compute_ipr_all` result to JSON.

    Parameters
    ----------
    result : dict
    outfile : str or Path
    """

    def _ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        return obj

    with open(outfile, "w") as fh:
        json.dump(_ser(result), fh, indent=2)
    n = len(result.get("band_summary", []))
    print(f"IPR JSON written to: {outfile}  ({n} bands)")


def save_ipr_csv(
    result: dict,
    outfile: Union[str, Path] = "ipr_summary.csv",
) -> None:
    """Save band-level IPR summary to CSV.

    Columns: iband, avg_energy, avg_ipr, avg_pr, avg_pr_norm,
             weighted_avg_ipr, weighted_avg_pr, weighted_avg_pr_norm,
             min_ipr, max_ipr, std_ipr

    Parameters
    ----------
    result : dict — from :func:`compute_ipr_all`.
    outfile : str or Path
    """
    fieldnames = [
        "iband",
        "avg_energy",
        "avg_ipr",
        "avg_pr",
        "avg_pr_norm",
        "weighted_avg_ipr",
        "weighted_avg_pr",
        "weighted_avg_pr_norm",
        "min_ipr",
        "max_ipr",
        "std_ipr",
    ]
    with open(outfile, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["band_summary"]:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"IPR CSV written to: {outfile}")


# ===========================================================================
# Optical properties (ZPL, Einstein A, radiative lifetime)
# ===========================================================================


def get_zpl(
    g_path: Union[str, Path],
    e_path: Union[str, Path],
    prefer: str = "oszicar",
) -> float:
    """Compute the zero-phonon line energy.

    ZPL = E_excited(relaxed) − E_ground(relaxed)

    Parameters
    ----------
    g_path : str or Path — ground-state calculation directory.
    e_path : str or Path — excited-state calculation directory.
    prefer : {'oszicar', 'outcar', 'vasprun'}

    Returns
    -------
    float — ZPL energy in eV.

    Examples
    --------
    >>> zpl = get_zpl("/data/ground/", "/data/excited/")
    """
    from defectpl.io.wavecar import get_total_energy

    return get_total_energy(e_path, prefer=prefer) - get_total_energy(
        g_path, prefer=prefer
    )


def get_dQ(
    g_path: Union[str, Path],
    e_path: Union[str, Path],
) -> float:
    """Mass-weighted atomic displacement dQ between ground and excited states.

    .. math::

        dQ = \\sqrt{\\sum_i m_i \\, |\\mathbf{r}_i^{(e)} - \\mathbf{r}_i^{(g)}|^2}

    Parameters
    ----------
    g_path : str or Path
    e_path : str or Path

    Returns
    -------
    float — dQ in amu^{1/2} · Å.

    Examples
    --------
    >>> dQ = get_dQ("/data/ground/", "/data/excited/")
    """
    from defectpl.io.wavecar import get_structure

    sg = get_structure(g_path, relaxed=True)
    se = get_structure(e_path, relaxed=True)
    if sg["natoms"] != se["natoms"]:
        raise ValueError(
            f"Atom count mismatch: ground={sg['natoms']}, excited={se['natoms']}"
        )
    pos_g = np.dot(sg["positions"], sg["lattice"])
    pos_e = np.dot(se["positions"], se["lattice"])
    masses = np.array(sg["masses"])
    delta = pos_e - pos_g
    return float(np.sqrt(np.sum(masses[:, None] * delta**2)))


def get_einstein_coefficient(
    zpl_ev: float,
    tdm_debye: float,
    nr: float = 2.42,
) -> float:
    """Einstein A coefficient (spontaneous emission rate) in MHz.

    .. math::

        A = \\frac{n_r E_{\\rm ZPL}^3 \\mu^2}{3 \\pi \\varepsilon_0 c^3 \\hbar^4}

    Parameters
    ----------
    zpl_ev : float — ZPL energy in eV.
    tdm_debye : float — |TDM| in Debye.
    nr : float — refractive index of the host.

    Returns
    -------
    float — A in MHz.

    Examples
    --------
    >>> A  = get_einstein_coefficient(1.95, 1.5, 2.42)
    >>> lt = get_radiative_lifetime(A)
    """
    eps0 = 55.26349406  # e² / (eV·µm)
    hbar = 6.5821e-16  # eV·s
    c = 299792458.0e6  # µm/s
    a0 = 5.29e-5  # Bohr radius in µm
    D2au = 0.393456  # Debye → e·a₀
    mu_um = tdm_debye * D2au * a0
    A = (nr * zpl_ev**3 * mu_um**2) / (3.0 * np.pi * eps0 * c**3 * hbar**4)
    return A * 1e-6  # Hz → MHz


def get_radiative_lifetime(A_mhz: float) -> float:
    """Convert Einstein A coefficient (MHz) to radiative lifetime (ns).

    Parameters
    ----------
    A_mhz : float

    Returns
    -------
    float — lifetime in ns.
    """
    return 1e3 / A_mhz


def compute_optical_properties(
    g_path: Union[str, Path],
    e_path: Union[str, Path],
    tdm_gg: Optional[float] = None,
    tdm_ge: Optional[float] = None,
    nr: float = 2.42,
    prefer_energy: str = "oszicar",
) -> dict:
    """Full optical property workflow for a defect colour centre.

    Parameters
    ----------
    g_path : str or Path — ground-state directory.
    e_path : str or Path — excited-state directory.
    tdm_gg : float, optional — same-state BZ-averaged |TDM| in Debye.
    tdm_ge : float, optional — cross-state |TDM| in Debye.
    nr : float — refractive index.
    prefer_energy : str — file type priority for energy reading.

    Returns
    -------
    dict with keys:
        ``"ZPL"``, ``"dQ"``, ``"E_ground"``, ``"E_excited"``, ``"nr"``,
        ``"tdm_gg"``, ``"tdm_ge"``,
        ``"A_gg"``, ``"A_ge"``,
        ``"lifetime_gg"``, ``"lifetime_ge"``.

    Examples
    --------
    >>> props = compute_optical_properties(
    ...     "/data/gs/", "/data/es/",
    ...     tdm_gg=1.5, tdm_ge=1.3, nr=2.65
    ... )
    >>> print(props["ZPL"], props["lifetime_gg"])
    """
    from defectpl.io.wavecar import get_total_energy

    E_g = get_total_energy(g_path, prefer=prefer_energy)
    E_e = get_total_energy(e_path, prefer=prefer_energy)
    zpl = E_e - E_g

    try:
        dQ = get_dQ(g_path, e_path)
    except Exception as exc:
        dQ = None
        print(f"  [warning] Could not compute dQ: {exc}")

    result: dict = {
        "E_ground": E_g,
        "E_excited": E_e,
        "ZPL": zpl,
        "dQ": dQ,
        "nr": nr,
        "tdm_gg": tdm_gg,
        "tdm_ge": tdm_ge,
        "A_gg": None,
        "A_ge": None,
        "lifetime_gg": None,
        "lifetime_ge": None,
    }
    if tdm_gg is not None:
        A = get_einstein_coefficient(zpl, tdm_gg, nr)
        result["A_gg"] = A
        result["lifetime_gg"] = get_radiative_lifetime(A)
    if tdm_ge is not None:
        A = get_einstein_coefficient(zpl, tdm_ge, nr)
        result["A_ge"] = A
        result["lifetime_ge"] = get_radiative_lifetime(A)
    return result
