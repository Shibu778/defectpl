# -*- coding: utf-8 -*-
"""
Tests for defectpl.physics.tdm — WavecarReader, IPR, select_bands, optical properties.

All WAVECAR-level tests use a minimal synthetic binary WAVECAR built in-memory
so no real VASP run is required.  The helper `_make_fake_wavecar` produces a
valid minimal binary file with 1 spin, 2 k-points, 4 bands, and random (but
deterministic) plane-wave coefficients.

Physical-constant helpers and select_bands are tested with mocked WavecarReader
objects (plain namespaces with the expected attributes) so the tests remain fast
and dependency-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defectpl.physics.tdm import (
    AUTOA,
    AUTDEBYE,
    RYTOEV,
    HSQDTM,
    TPI,
    WavecarReader,
    select_bands,
    compute_ipr_band,
    compute_ipr_all,
    compute_ipr_weighted,
    save_ipr_json,
    save_ipr_csv,
    get_zpl,
    get_einstein_coefficient,
    get_radiative_lifetime,
    compute_optical_properties,
    get_weighted_avg_tdm,
    _select_band_pairs_at_kpoint,
)


def _compute_nplw(
    Acell: np.ndarray, Bcell: np.ndarray, kvec: np.ndarray, encut: float
) -> int:
    """Compute the number of G-vectors that pass the ENCUT cutoff.

    Replicates the logic in WavecarReader.gvectors() so that synthetic
    WAVECARs have nplw consistent with what the reader expects.
    """
    from math import ceil, sqrt

    Anorm = np.linalg.norm(Acell, axis=1)
    cutof = np.array([ceil(sqrt(encut / RYTOEV) / (TPI / (a / AUTOA))) for a in Anorm])
    ngrid = np.array(2 * cutof + 1, dtype=int)

    fx = np.arange(ngrid[0], dtype=int)
    fx[ngrid[0] // 2 + 1 :] -= ngrid[0]
    fy = np.arange(ngrid[1], dtype=int)
    fy[ngrid[1] // 2 + 1 :] -= ngrid[1]
    fz = np.arange(ngrid[2], dtype=int)
    fz[ngrid[2] // 2 + 1 :] -= ngrid[2]

    gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape(3, -1)
    kgrid = np.column_stack([gx, gy, gz]).astype(float)
    ke = (
        HSQDTM
        * np.linalg.norm(np.dot(kgrid + kvec[np.newaxis, :], TPI * Bcell), axis=1) ** 2
    )
    return int(np.sum(ke < encut))


# ===========================================================================
# Synthetic WAVECAR builder
# ===========================================================================


def _pack_record(data: np.ndarray, recl: int) -> bytes:
    raw = data.tobytes()
    padded = raw + b"\x00" * (recl - len(raw))
    return padded[:recl]


def _make_fake_wavecar(
    tmp_path: Path,
    nspin: int = 1,
    nkpts: int = 2,
    nbands: int = 4,
    encut: float = 200.0,
    rtag: int = 45200,  # complex64
    seed: int = 42,
) -> Path:
    """Write a minimal but valid complex64 WAVECAR to *tmp_path/WAVECAR*."""
    rng = np.random.default_rng(seed)

    # Lattice: simple cubic 5 Å
    Acell = np.diag([5.0, 5.0, 5.0])
    Bcell = np.linalg.inv(Acell).T

    kvecs = rng.uniform(-0.5, 0.5, size=(nkpts, 3))

    # Compute nplw exactly as WavecarReader.gvectors() does to avoid mismatch
    gvec_sets = [_compute_nplw(Acell, Bcell, kv, encut) for kv in kvecs]

    # recl must be large enough for the biggest coefficient record
    # complex64 = 8 bytes/coefficient; add 1024 bytes padding for headers
    max_nplw = max(gvec_sets) if gvec_sets else 1
    recl = max(3200, max_nplw * 8 + 1024)

    eigenvalues = rng.uniform(-5.0, 5.0, size=(nspin, nkpts, nbands))
    occupancies = np.zeros((nspin, nkpts, nbands))
    for s in range(nspin):
        for k in range(nkpts):
            occs_k = [1.0 if eigenvalues[s, k, b] < 0.0 else 0.0 for b in range(nbands)]
            occupancies[s, k, :] = occs_k

    wavecar_path = tmp_path / "WAVECAR"
    with open(wavecar_path, "wb") as fout:
        # Record 0 — header
        r0 = np.array([recl, nspin, rtag], dtype=np.float64)
        fout.write(_pack_record(r0, recl))

        # Record 1 — nkpts / nbands / encut / Acell
        r1 = np.concatenate(
            [
                [float(nkpts), float(nbands), encut],
                Acell.flatten(),
            ]
        ).astype(np.float64)
        fout.write(_pack_record(r1, recl))

        for ispin in range(nspin):
            for ikpt in range(nkpts):
                nplw = gvec_sets[ikpt]
                kv = kvecs[ikpt]
                kpt_header = np.zeros(4 + 3 * nbands, dtype=np.float64)
                kpt_header[0] = nplw
                kpt_header[1:4] = kv
                for b in range(nbands):
                    kpt_header[4 + 3 * b] = eigenvalues[ispin, ikpt, b]
                    kpt_header[4 + 3 * b + 1] = 0.0
                    kpt_header[4 + 3 * b + 2] = occupancies[ispin, ikpt, b]
                fout.write(_pack_record(kpt_header, recl))

                for b in range(nbands):
                    cg = (
                        rng.standard_normal(nplw) + 1j * rng.standard_normal(nplw)
                    ).astype(np.complex64)
                    cg /= np.linalg.norm(cg)
                    fout.write(_pack_record(cg.view(np.float32), recl))

    return wavecar_path


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def fake_wavecar_path(tmp_path_factory):
    return _make_fake_wavecar(tmp_path_factory.mktemp("wavecar"))


@pytest.fixture(scope="module")
def wfc(fake_wavecar_path):
    return WavecarReader(str(fake_wavecar_path))


@pytest.fixture()
def mock_wfc():
    """Minimal WavecarReader-like namespace for select_bands / IPR tests."""
    nkpts = 3
    nbands = 6
    rng = np.random.default_rng(7)
    # eigenvalues spread from -3 to 3 eV, first 3 bands negative (occupied)
    bands = np.array([np.linspace(-3.0, 3.0, nbands) for _ in range(nkpts)])[
        np.newaxis, ...
    ]  # shape (1, nkpts, nbands)
    occs = np.where(bands < 0.0, 1.0, 0.0)
    kvecs = rng.uniform(-0.5, 0.5, size=(nkpts, 3))
    Acell = np.diag([5.0, 5.0, 5.0])
    Bcell = np.linalg.inv(Acell).T

    def wfc_r(ispin, ikpt, iband, ngrid=None):
        rng2 = np.random.default_rng(ispin * 1000 + ikpt * 100 + iband)
        phi_r = rng2.standard_normal((4, 4, 4)) + 1j * rng2.standard_normal((4, 4, 4))
        return phi_r

    ns = SimpleNamespace(
        nspin=1,
        nkpts=nkpts,
        nbands=nbands,
        bands=bands,
        occs=occs,
        kvecs=kvecs,
        Acell=Acell,
        Bcell=Bcell,
        wfc_r=wfc_r,
    )
    return ns


# ===========================================================================
# Physical constants sanity
# ===========================================================================


class TestPhysicalConstants:
    def test_autoa(self):
        assert pytest.approx(AUTOA, rel=1e-5) == 0.529177249

    def test_rytoev(self):
        assert pytest.approx(RYTOEV, rel=1e-5) == 13.605826

    def test_autdebye(self):
        assert pytest.approx(AUTDEBYE, rel=1e-4) == 2.541746

    def test_hsqdtm_positive(self):
        assert HSQDTM > 0


# ===========================================================================
# WavecarReader — header parsing
# ===========================================================================


class TestWavecarReaderHeader:
    def test_nspin(self, wfc):
        assert wfc.nspin == 1

    def test_nkpts(self, wfc):
        assert wfc.nkpts == 2

    def test_nbands(self, wfc):
        assert wfc.nbands == 4

    def test_encut(self, wfc):
        assert wfc.encut == pytest.approx(200.0, rel=1e-4)

    def test_acell_shape(self, wfc):
        assert wfc.Acell.shape == (3, 3)

    def test_bcell_inverse_relation(self, wfc):
        product = np.dot(wfc.Acell, wfc.Bcell.T)
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_kvecs_shape(self, wfc):
        assert wfc.kvecs.shape == (2, 3)

    def test_bands_shape(self, wfc):
        assert wfc.bands.shape == (1, 2, 4)

    def test_occs_shape(self, wfc):
        assert wfc.occs.shape == (1, 2, 4)


# ===========================================================================
# WavecarReader — G-vector generation
# ===========================================================================


class TestGvectors:
    def test_returns_array(self, wfc):
        G = wfc.gvectors(1)
        assert isinstance(G, np.ndarray)

    def test_shape_second_axis(self, wfc):
        G = wfc.gvectors(1)
        assert G.ndim == 2
        assert G.shape[1] == 3

    def test_dtype_int(self, wfc):
        G = wfc.gvectors(1)
        assert np.issubdtype(G.dtype, np.integer)

    def test_ikpt_out_of_range(self, wfc):
        with pytest.raises(IndexError):
            wfc.gvectors(99)


# ===========================================================================
# WavecarReader — coefficient reading
# ===========================================================================


class TestReadBandCoeff:
    def test_returns_complex_array(self, wfc):
        cg = wfc.read_band_coeff(1, 1, 1)
        assert np.iscomplexobj(cg)

    def test_length_matches_nplw(self, wfc):
        cg = wfc.read_band_coeff(1, 1, 1)
        nplw = wfc._nplws[0]
        assert len(cg) == nplw

    def test_norm_flag(self, wfc):
        cg = wfc.read_band_coeff(1, 1, 1, norm=True)
        assert pytest.approx(np.linalg.norm(cg), abs=1e-6) == 1.0

    def test_out_of_range_ispin(self, wfc):
        with pytest.raises(IndexError):
            wfc.read_band_coeff(3, 1, 1)

    def test_out_of_range_ikpt(self, wfc):
        with pytest.raises(IndexError):
            wfc.read_band_coeff(1, 99, 1)

    def test_out_of_range_iband(self, wfc):
        with pytest.raises(IndexError):
            wfc.read_band_coeff(1, 1, 99)


# ===========================================================================
# WavecarReader — TDM
# ===========================================================================


class TestGetMomentumMatrix:
    def test_shape(self, wfc):
        mom = wfc.get_momentum_matrix(1, 1, 1, 2)
        assert mom.shape == (3,)

    def test_complex(self, wfc):
        mom = wfc.get_momentum_matrix(1, 1, 1, 2)
        assert np.iscomplexobj(mom)


class TestGetDipoleMatrix:
    def test_returns_four_values(self, wfc):
        result = wfc.get_dipole_matrix(1, 1, 1, 2)
        assert len(result) == 4

    def test_dipole_shape(self, wfc):
        _, _, _, dip = wfc.get_dipole_matrix(1, 1, 1, 2)
        assert dip.shape == (3,)

    def test_zero_dE_returns_zero_dipole(self, wfc):
        # Same band index → ΔE = 0
        _, _, dE, dip = wfc.get_dipole_matrix(1, 1, 1, 1)
        assert pytest.approx(dE, abs=1e-12) == 0.0
        np.testing.assert_array_almost_equal(np.abs(dip), [0.0, 0.0, 0.0])


class TestGetTdmAllKpoints:
    def test_keys(self, wfc):
        res = wfc.get_tdm_all_kpoints(1, 1, 2)
        for k in (
            "ispin",
            "iband_i",
            "iband_j",
            "kvecs",
            "E_i",
            "E_j",
            "dE",
            "tdm_components",
            "tdm_magnitude",
        ):
            assert k in res

    def test_nkpts_dimension(self, wfc):
        res = wfc.get_tdm_all_kpoints(1, 1, 2)
        assert len(res["tdm_magnitude"]) == wfc.nkpts
        assert res["tdm_components"].shape == (wfc.nkpts, 3)

    def test_magnitude_non_negative(self, wfc):
        res = wfc.get_tdm_all_kpoints(1, 1, 2)
        assert np.all(res["tdm_magnitude"] >= 0.0)

    def test_ispin_iband_stored(self, wfc):
        res = wfc.get_tdm_all_kpoints(1, 1, 3)
        assert res["ispin"] == 1
        assert res["iband_i"] == 1
        assert res["iband_j"] == 3


class TestGetWeightedAvgTdm:
    def test_shape(self, wfc):
        kw = np.ones(wfc.nkpts)
        res = wfc.get_weighted_avg_tdm(1, 1, 2, kw)
        assert res["avg_tdm_components"].shape == (3,)

    def test_magnitude_positive(self, wfc):
        kw = np.ones(wfc.nkpts)
        res = wfc.get_weighted_avg_tdm(1, 1, 2, kw)
        assert res["avg_tdm_magnitude"] >= 0.0

    def test_wrong_kweights_length(self, wfc):
        with pytest.raises(ValueError):
            wfc.get_weighted_avg_tdm(1, 1, 2, np.ones(99))


# ===========================================================================
# get_weighted_avg_tdm (free function)
# ===========================================================================


class TestGetWeightedAvgTdmFn:
    def test_uniform_weights(self):
        tdm_per_kpt = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        kw = np.array([1.0, 1.0])
        avg = get_weighted_avg_tdm(tdm_per_kpt, kw)
        np.testing.assert_allclose(avg, [1.0, 0.0, 0.0])

    def test_biased_weights(self):
        tdm_per_kpt = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        kw = np.array([1.0, 0.0])
        avg = get_weighted_avg_tdm(tdm_per_kpt, kw)
        np.testing.assert_allclose(avg, [2.0, 0.0, 0.0])


# ===========================================================================
# WavecarReader — compact WAVECAR save/reload
# ===========================================================================


class TestCompactWavecar:
    def test_compact_roundtrip(self, fake_wavecar_path, tmp_path):
        wfc = WavecarReader(str(fake_wavecar_path))
        compact = tmp_path / "WAVECAR_compact"
        kept = wfc.save_compact_wavecar([1, 2], outfile=compact)
        assert sorted(kept) == [1, 2]

        wfc2 = WavecarReader(str(compact))
        assert wfc2.nbands == wfc.nbands  # original nbands recovered
        # Coefficients of band 1 in original and compact should match
        cg_orig = wfc.read_band_coeff(1, 1, 1)
        cg_compact = wfc2.read_band_coeff(1, 1, 1)
        np.testing.assert_allclose(cg_orig, cg_compact, atol=1e-6)

    def test_compact_missing_band_raises(self, fake_wavecar_path, tmp_path):
        wfc = WavecarReader(str(fake_wavecar_path))
        compact = tmp_path / "WAVECAR_c2"
        wfc.save_compact_wavecar([1, 2], outfile=compact)
        wfc2 = WavecarReader(str(compact))
        with pytest.raises((KeyError, IndexError)):
            wfc2.read_band_coeff(1, 1, 4)  # band 4 not stored


class TestTrimWavecar:
    def test_trim_roundtrip(self, fake_wavecar_path, tmp_path):
        wfc = WavecarReader(str(fake_wavecar_path))
        trimmed = tmp_path / "WAVECAR_trim"
        kept = wfc.trim_save_wavecar([1, 3], outfile=trimmed)
        assert sorted(kept) == [1, 3]

        wfc2 = WavecarReader(str(trimmed))
        assert wfc2.nbands == wfc.nbands  # full nbands preserved
        # Coefficients of stored band should match
        cg_orig = wfc.read_band_coeff(1, 1, 1)
        cg_trim = wfc2.read_band_coeff(1, 1, 1)
        np.testing.assert_allclose(cg_orig, cg_trim, atol=1e-6)


# ===========================================================================
# WavecarReader — JSON export
# ===========================================================================


class TestSaveToJson:
    def test_creates_file(self, wfc, tmp_path):
        out = tmp_path / "info.json"
        wfc.save_to_json(bands=[1, 2], outfile=out)
        assert out.exists()

    def test_json_structure(self, wfc, tmp_path):
        out = tmp_path / "info.json"
        wfc.save_to_json(bands=[1], outfile=out)
        with open(out) as fh:
            data = json.load(fh)
        assert "nkpts" in data
        assert "eigenvalues" in data

    def test_bands_selected_field(self, wfc, tmp_path):
        out = tmp_path / "info2.json"
        wfc.save_to_json(bands=[1, 3], outfile=out)
        with open(out) as fh:
            data = json.load(fh)
        assert data["bands_selected"] == [1, 3]


# ===========================================================================
# WavecarReader — wfc_r real-space wavefunction
# ===========================================================================


class TestWfcR:
    def test_returns_3d_array(self, wfc):
        phi = wfc.wfc_r(1, 1, 1)
        assert phi.ndim == 3

    def test_complex(self, wfc):
        phi = wfc.wfc_r(1, 1, 1)
        assert np.iscomplexobj(phi)

    def test_custom_ngrid(self, wfc):
        phi = wfc.wfc_r(1, 1, 1, ngrid=[8, 8, 8])
        assert phi.shape == (8, 8, 8)


# ===========================================================================
# select_bands
# ===========================================================================


class TestSelectBands:
    def test_all_mode(self, mock_wfc):
        bands = select_bands(mock_wfc, 1, mode="all")
        assert bands == list(range(1, mock_wfc.nbands + 1))

    def test_band_list_mode(self, mock_wfc):
        bands = select_bands(mock_wfc, 1, mode="band_list", band_list=[1, 3, 5])
        assert bands == [1, 3, 5]

    def test_band_list_missing_raises(self, mock_wfc):
        with pytest.raises(ValueError, match="band_list"):
            select_bands(mock_wfc, 1, mode="band_list")

    def test_band_range_mode(self, mock_wfc):
        bands = select_bands(mock_wfc, 1, mode="band_range", band_range=(2, 4))
        assert bands == [2, 3, 4]

    def test_band_range_missing_raises(self, mock_wfc):
        with pytest.raises(ValueError, match="band_range"):
            select_bands(mock_wfc, 1, mode="band_range")

    def test_occupation_mode(self, mock_wfc):
        bands = select_bands(mock_wfc, 1, mode="occupation")
        assert len(bands) > 0

    def test_near_fermi_count(self, mock_wfc):
        bands = select_bands(mock_wfc, 1, mode="near_fermi", n_occ=2, n_unocc=2)
        assert len(bands) <= 4

    def test_homo_lumo_range(self, mock_wfc):
        bands = select_bands(
            mock_wfc, 1, mode="homo_lumo_range", below_homo_ev=1.0, above_lumo_ev=1.0
        )
        assert len(bands) > 0

    def test_energy_window_mode(self, mock_wfc):
        bands = select_bands(
            mock_wfc, 1, mode="energy_window", energy_min=-2.0, energy_max=2.0
        )
        assert len(bands) > 0

    def test_energy_window_missing_raises(self, mock_wfc):
        with pytest.raises(ValueError, match="energy_min"):
            select_bands(mock_wfc, 1, mode="energy_window")

    def test_unknown_mode_raises(self, mock_wfc):
        with pytest.raises(ValueError, match="Unknown mode"):
            select_bands(mock_wfc, 1, mode="nonsense")

    def test_with_fermi_level(self, mock_wfc):
        bands = select_bands(
            mock_wfc, 1, mode="near_fermi", fermi_level=0.0, n_occ=2, n_unocc=2
        )
        assert isinstance(bands, list)


# ===========================================================================
# _select_band_pairs_at_kpoint
# ===========================================================================


class TestSelectBandPairsAtKpoint:
    @pytest.fixture()
    def arrays(self):
        nbands = 6
        nkpts = 2
        bands = np.linspace(-3.0, 3.0, nbands)[None, None, :] * np.ones((1, nkpts, 1))
        occs = np.where(bands < 0.0, 1.0, 0.0)
        return bands, occs

    def test_occupation_mode(self, arrays):
        bands, occs = arrays
        occ, unocc = _select_band_pairs_at_kpoint(
            bands, occs, 1, 1, "occupation", occ_threshold=0.5
        )
        assert all(b in occ for b in [1, 2, 3])
        assert all(b in unocc for b in [4, 5, 6])

    def test_band_range_mode(self, arrays):
        bands, occs = arrays
        occ, unocc = _select_band_pairs_at_kpoint(
            bands,
            occs,
            1,
            1,
            "band_range",
            occ_bands=(1, 3),
            unocc_bands=(4, 6),
        )
        assert occ == [1, 2, 3]
        assert unocc == [4, 5, 6]

    def test_band_list_mode(self, arrays):
        bands, occs = arrays
        occ, unocc = _select_band_pairs_at_kpoint(
            bands,
            occs,
            1,
            1,
            "band_list",
            occ_band_list=[1, 2],
            unocc_band_list=[4, 5],
        )
        assert occ == [1, 2]
        assert unocc == [4, 5]

    def test_unknown_mode_raises(self, arrays):
        bands, occs = arrays
        with pytest.raises(ValueError):
            _select_band_pairs_at_kpoint(bands, occs, 1, 1, "bad_mode")


# ===========================================================================
# IPR functions
# ===========================================================================


class TestComputeIprBand:
    def test_returns_three_values(self, mock_wfc):
        ipr, pr, pr_norm = compute_ipr_band(mock_wfc, 1, 1, 1)
        assert isinstance(ipr, float)
        assert isinstance(pr, float)
        assert isinstance(pr_norm, float)

    def test_ipr_positive(self, mock_wfc):
        ipr, _, _ = compute_ipr_band(mock_wfc, 1, 1, 1)
        assert ipr > 0

    def test_pr_norm_in_unit_interval(self, mock_wfc):
        _, _, pr_norm = compute_ipr_band(mock_wfc, 1, 1, 1)
        assert 0.0 <= pr_norm <= 1.0

    def test_ipr_pr_inverse(self, mock_wfc):
        ipr, pr, _ = compute_ipr_band(mock_wfc, 1, 1, 1)
        assert pytest.approx(ipr * pr, rel=1e-6) == 1.0


class TestComputeIprAll:
    def test_has_expected_keys(self, mock_wfc):
        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2], verbose=False)
        for k in ("metadata", "per_band_per_kpoint", "band_summary", "kweights"):
            assert k in result

    def test_band_summary_length(self, mock_wfc):
        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2, 3], verbose=False)
        assert len(result["band_summary"]) == 3

    def test_kweights_default_uniform(self, mock_wfc):
        result = compute_ipr_all(mock_wfc, 1, bands=[1], verbose=False)
        np.testing.assert_array_equal(result["kweights"], np.ones(mock_wfc.nkpts))

    def test_select_mode_all(self, mock_wfc):
        result = compute_ipr_all(mock_wfc, 1, select_mode="all", verbose=False)
        assert len(result["band_summary"]) == mock_wfc.nbands

    def test_per_band_per_kpoint_count(self, mock_wfc):
        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2], verbose=False)
        assert len(result["per_band_per_kpoint"]) == 2 * mock_wfc.nkpts


class TestComputeIprWeighted:
    def test_has_expected_keys(self, mock_wfc):
        result = compute_ipr_weighted(mock_wfc, 1, 1)
        for k in (
            "avg_ipr",
            "weighted_avg_ipr",
            "avg_pr",
            "weighted_avg_pr",
            "avg_pr_norm",
            "weighted_avg_pr_norm",
        ):
            assert k in result

    def test_weighted_avg_ipr_positive(self, mock_wfc):
        result = compute_ipr_weighted(mock_wfc, 1, 1)
        assert result["weighted_avg_ipr"] > 0


class TestSaveIprFiles:
    def test_json_roundtrip(self, mock_wfc, tmp_path):
        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2], verbose=False)
        outfile = tmp_path / "ipr.json"
        save_ipr_json(result, outfile)
        with open(outfile) as fh:
            data = json.load(fh)
        assert "band_summary" in data
        assert len(data["band_summary"]) == 2

    def test_csv_created(self, mock_wfc, tmp_path):
        result = compute_ipr_all(mock_wfc, 1, bands=[1], verbose=False)
        outfile = tmp_path / "ipr.csv"
        save_ipr_csv(result, outfile)
        assert outfile.exists()
        content = outfile.read_text()
        assert "iband" in content


# ===========================================================================
# Optical property helpers
# ===========================================================================


class TestGetEinsteinCoefficient:
    def test_positive(self):
        A = get_einstein_coefficient(1.95, 1.5, 2.42)
        assert A > 0

    def test_lifetime_positive(self):
        A = get_einstein_coefficient(1.95, 1.5, 2.42)
        lt = get_radiative_lifetime(A)
        assert lt > 0

    def test_lifetime_inverse_relation(self):
        A = get_einstein_coefficient(1.95, 1.5, 2.42)
        lt = get_radiative_lifetime(A)
        assert pytest.approx(A * lt, rel=1e-6) == 1e3  # MHz × ns = 1

    def test_scales_with_tdm_squared(self):
        A1 = get_einstein_coefficient(2.0, 1.0, 1.0)
        A2 = get_einstein_coefficient(2.0, 2.0, 1.0)
        assert pytest.approx(A2 / A1, rel=1e-5) == 4.0

    def test_scales_with_nr(self):
        A1 = get_einstein_coefficient(2.0, 1.0, 1.0)
        A2 = get_einstein_coefficient(2.0, 1.0, 2.0)
        assert pytest.approx(A2 / A1, rel=1e-5) == 2.0


class TestComputeOpticalProperties:
    def test_returns_dict(self, tmp_path):
        from textwrap import dedent

        OSZICAR = dedent("""\
            N       E                     dE             d eps       ncg     rms          rms(c)
                  1 F= -.54321000E+02 E0= -.54321000E+02  d E =-.54321000E+02
        """)
        OSZICAR_E = dedent("""\
            N       E                     dE             d eps       ncg     rms          rms(c)
                  1 F= -.54200000E+02 E0= -.54200000E+02  d E =-.54200000E+02
        """)
        g_dir = tmp_path / "gs"
        e_dir = tmp_path / "es"
        g_dir.mkdir()
        e_dir.mkdir()
        (g_dir / "OSZICAR").write_text(OSZICAR)
        (e_dir / "OSZICAR").write_text(OSZICAR_E)

        POSCAR = dedent("""\
            Si
              1.0
                3.86746  0.00000  0.00000
                0.00000  3.86746  0.00000
                0.00000  0.00000  3.86746
            Si
            2
            Direct
              0.00000  0.00000  0.00000
              0.25000  0.25000  0.25000
        """)
        (g_dir / "POSCAR").write_text(POSCAR)
        (e_dir / "POSCAR").write_text(POSCAR)

        props = compute_optical_properties(
            g_path=g_dir,
            e_path=e_dir,
            tdm_gg=1.5,
            nr=2.42,
        )
        assert "ZPL" in props
        assert "A_gg" in props
        assert "lifetime_gg" in props
        assert props["A_gg"] > 0
        assert props["lifetime_gg"] > 0

    def test_zpl_sign(self, tmp_path):
        from textwrap import dedent

        OSZICAR_GS = dedent("""\
            N   E   dE   d eps   ncg rms   rms(c)
              1 F= -.54321000E+02 E0= -.54321000E+02  d E =-.54321000E+02
        """)
        OSZICAR_ES = dedent("""\
            N   E   dE   d eps   ncg rms   rms(c)
              1 F= -.54000000E+02 E0= -.54000000E+02  d E =-.54000000E+02
        """)
        g_dir = tmp_path / "gs2"
        e_dir = tmp_path / "es2"
        g_dir.mkdir()
        e_dir.mkdir()
        (g_dir / "OSZICAR").write_text(OSZICAR_GS)
        (e_dir / "OSZICAR").write_text(OSZICAR_ES)

        zpl = get_zpl(g_dir, e_dir)
        assert zpl > 0  # excited state higher in energy


# ===========================================================================
# TDM visualisation (smoke tests — just check no exception)
# ===========================================================================


class TestTdmViz:
    @pytest.fixture()
    def tdm_result(self, wfc):
        return wfc.get_tdm_all_kpoints(1, 1, 2)

    def test_plot_tdm_heatmap(self, tdm_result, tmp_path):
        from defectpl.physics.tdm_viz import plot_tdm_heatmap

        out = tmp_path / "heatmap.png"
        fig = plot_tdm_heatmap(tdm_result, outfile=out)
        assert fig is not None
        assert out.exists()

    def test_plot_tdm_bubble(self, tdm_result, tmp_path):
        from defectpl.physics.tdm_viz import plot_tdm_bubble

        out = tmp_path / "bubble.png"
        plot_tdm_bubble(tdm_result, outfile=out)
        assert out.exists()

    def test_plot_tdm_components(self, tdm_result, tmp_path):
        from defectpl.physics.tdm_viz import plot_tdm_components

        out = tmp_path / "comp.png"
        plot_tdm_components(tdm_result, outfile=out)
        assert out.exists()

    def test_plot_tdm_absorption(self, tdm_result, tmp_path):
        from defectpl.physics.tdm_viz import plot_tdm_absorption

        out = tmp_path / "abs.png"
        _, E, spec = plot_tdm_absorption(tdm_result, outfile=out)
        assert len(E) == len(spec)
        assert out.exists()

    def test_plot_tdm_dashboard(self, tdm_result, tmp_path):
        from defectpl.physics.tdm_viz import plot_tdm_dashboard

        out = tmp_path / "dash.png"
        plot_tdm_dashboard(tdm_result, outfile=out)
        assert out.exists()


class TestIprViz:
    def test_plot_ipr_scatter(self, mock_wfc, tmp_path):
        from defectpl.physics.tdm_viz import plot_ipr_scatter

        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2, 3], verbose=False)
        out = tmp_path / "scatter.png"
        plot_ipr_scatter(result, outfile=out)
        assert out.exists()

    def test_plot_ipr_bar(self, mock_wfc, tmp_path):
        from defectpl.physics.tdm_viz import plot_ipr_bar

        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2, 3], verbose=False)
        out = tmp_path / "bar.png"
        plot_ipr_bar(result, top_n=3, outfile=out)
        assert out.exists()

    def test_plot_ipr_kpoint_heatmap(self, mock_wfc, tmp_path):
        from defectpl.physics.tdm_viz import plot_ipr_kpoint_heatmap

        result = compute_ipr_all(mock_wfc, 1, bands=[1, 2], verbose=False)
        out = tmp_path / "ipr_heatmap.png"
        plot_ipr_kpoint_heatmap(result, outfile=out)
        assert out.exists()


class TestWfcExport:
    @pytest.fixture()
    def structure(self):
        return {
            "title": "test",
            "scale": 1.0,
            "lattice": np.diag([5.0, 5.0, 5.0]),
            "species": ["Si"],
            "counts": [2],
            "atom_species": ["Si", "Si"],
            "positions": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            "masses": [28.086, 28.086],
            "natoms": 2,
        }

    def test_save_wfc_vasp_creates_file(self, wfc, structure, tmp_path):
        from defectpl.physics.tdm_viz import save_wfc_vasp

        out = tmp_path / "wfc.vasp"
        save_wfc_vasp(wfc, 1, 1, 1, structure, outfile=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_wfc_vesta_creates_file(self, wfc, structure, tmp_path):
        from defectpl.physics.tdm_viz import save_wfc_vasp, save_wfc_vesta

        vasp_f = tmp_path / "wfc.vasp"
        save_wfc_vasp(wfc, 1, 1, 1, structure, outfile=vasp_f)
        out = tmp_path / "wfc.vesta"
        save_wfc_vesta(wfc, 1, 1, 1, structure, vasp_file=vasp_f, outfile=out)
        assert out.exists()
        content = out.read_text()
        assert "VESTA_FORMAT_VERSION" in content
        assert "IMPORT_DENSITY" in content
