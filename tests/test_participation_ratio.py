"""
tests/test_participation_ratio.py
==================================

Unit and integration tests for ``defectpl.participation_ratio`` and
``defectpl.defect_utils``.

All tests use synthetic in-memory data so no real VASP files are required.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from defectpl.participation_ratio import (
    ParticipationRatioCalculator,
    compute_participation_ratios,
    neighbors_from_defect_structure_info,
    resolve_neighbors,
    _parse_procar_native,
)
from defectpl.defect_utils import (
    make_defect_entry,
    make_defect_structure_info,
    parse_frac_coords,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

N_IONS     = 8
N_KPOINTS  = 2
N_BANDS    = 4

NEIGHBOR_INDICES = [1, 3]


def _make_site_proj(seed: int = 42) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    proj = rng.uniform(0.0, 1.0, (N_KPOINTS, N_BANDS, N_IONS))
    proj /= proj.sum(axis=2, keepdims=True)
    return proj


def _spin_sentinel(value: int = 1):
    """Return a hashable spin sentinel compatible with the native parser."""
    class _Spin(int):
        pass
    return _Spin(value)


def _make_procar_data(seed: int = 42) -> dict:
    """Build a synthetic procar_data dict matching read_procar's output schema."""
    try:
        from pymatgen.electronic_structure.core import Spin
        spin_up = Spin.up
    except ImportError:
        spin_up = _spin_sentinel(1)

    proj   = _make_site_proj(seed)
    eigval = np.linspace(-5.0, 5.0, N_KPOINTS * N_BANDS).reshape(N_KPOINTS, N_BANDS)
    occ    = (eigval < 0.0).astype(float)

    return {
        "n_kpoints":   N_KPOINTS,
        "n_bands":     N_BANDS,
        "n_ions":      N_IONS,
        "n_spins":     1,
        "spins":       [spin_up],
        "site_proj":   {spin_up: proj},
        "eigenvalues": {spin_up: eigval},
        "occupancies": {spin_up: occ},
        "kpoints":     None,
        "weights":     None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1.  compute_participation_ratios
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeParticipationRatios:

    def setup_method(self):
        self.procar_data = _make_procar_data()

    def test_return_schema(self):
        result = compute_participation_ratios(
            self.procar_data, NEIGHBOR_INDICES,
            defect_name="Va_O1_2",
            defect_center=[0.5, 0.5, 0.5],
        )
        for key in ("defect_name", "defect_center",
                    "neighbor_atom_indices", "n_atoms",
                    "n_spins", "n_kpoints", "n_bands", "data"):
            assert key in result, f"Missing key: {key}"

    def test_dimensions(self):
        result  = compute_participation_ratios(self.procar_data, NEIGHBOR_INDICES)
        data    = result["data"]
        assert len(data) == 1
        sp_data = next(iter(data.values()))
        assert len(sp_data) == N_KPOINTS
        kpt_data = next(iter(sp_data.values()))
        assert len(kpt_data) == N_BANDS

    def test_p_ratio_range(self):
        result = compute_participation_ratios(self.procar_data, NEIGHBOR_INDICES)
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    pr = band_vals["p_ratio"]
                    assert 0.0 <= pr <= 1.0, f"P-ratio out of [0,1]: {pr}"

    def test_ipr_range(self):
        result = compute_participation_ratios(self.procar_data, NEIGHBOR_INDICES)
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    ipr = band_vals["ipr"]
                    assert 0.0 <= ipr <= 1.0 + 1e-9, f"IPR out of range: {ipr}"
                    assert ipr >= 1.0 / N_IONS - 1e-9, f"IPR below 1/N: {ipr}"

    def test_no_neighbors_gives_zero_p_ratio(self):
        result = compute_participation_ratios(self.procar_data, [])
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    assert band_vals["p_ratio"] == 0.0

    def test_all_neighbors_gives_unit_p_ratio(self):
        all_idx = list(range(N_IONS))
        result  = compute_participation_ratios(self.procar_data, all_idx)
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    assert math.isclose(band_vals["p_ratio"], 1.0, abs_tol=1e-6)

    def test_fully_localized_ipr(self):
        procar = _make_procar_data()
        spin   = procar["spins"][0]
        proj   = procar["site_proj"][spin]
        proj[:] = 0.0
        proj[:, :, 0] = 1.0
        result = compute_participation_ratios(procar, [0])
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    assert math.isclose(band_vals["ipr"], 1.0, abs_tol=1e-6)

    def test_uniform_projection_ipr(self):
        procar = _make_procar_data()
        spin   = procar["spins"][0]
        procar["site_proj"][spin][:] = 1.0 / N_IONS
        result = compute_participation_ratios(procar, [])
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    assert math.isclose(band_vals["ipr"], 1.0 / N_IONS, abs_tol=1e-6)

    def test_neighbor_atom_indices_sorted_deduplicated(self):
        result = compute_participation_ratios(self.procar_data, [3, 1, 3, 1])
        assert result["neighbor_atom_indices"] == [1, 3]

    def test_defect_name_propagated(self):
        result = compute_participation_ratios(
            self.procar_data, [], defect_name="Va_Mg1_-2"
        )
        assert result["defect_name"] == "Va_Mg1_-2"

    def test_energy_none_when_no_eigenvalues(self):
        procar = _make_procar_data()
        procar["eigenvalues"] = None
        result = compute_participation_ratios(procar, NEIGHBOR_INDICES)
        for sp_data in result["data"].values():
            for kpt_data in sp_data.values():
                for band_vals in kpt_data.values():
                    assert band_vals["energy"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Native PROCAR parser
# ──────────────────────────────────────────────────────────────────────────────

_PROCAR_MINIMAL = """\
PROCAR lm decomposed
# of k-points:    1         # of bands:    2         # of ions:    3

 k-point      1 :    0.00000000 0.00000000 0.00000000     weight = 1.00000000

band     1 # energy   -2.00000000 # occ.  1.00000000

ion      s     py     pz     px    tot
  1  0.100  0.050  0.000  0.000  0.150
  2  0.200  0.100  0.000  0.000  0.300
  3  0.050  0.050  0.000  0.000  0.100
tot  0.350  0.200  0.000  0.000  0.550

band     2 # energy    1.00000000 # occ.  0.00000000

ion      s     py     pz     px    tot
  1  0.010  0.000  0.000  0.000  0.010
  2  0.500  0.200  0.000  0.000  0.700
  3  0.100  0.050  0.000  0.000  0.150
tot  0.610  0.250  0.000  0.000  0.860

"""


class TestNativeProcarParser:

    def _write_procar(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "PROCAR"
        p.write_text(content, encoding="utf-8")
        return p

    def test_parse_minimal_procar(self, tmp_path: Path):
        p = self._write_procar(tmp_path, _PROCAR_MINIMAL)
        data = _parse_procar_native(p)
        assert data["n_kpoints"] == 1
        assert data["n_bands"]   == 2
        assert data["n_ions"]    == 3
        assert data["n_spins"]   == 1

    def test_site_proj_values(self, tmp_path: Path):
        p    = self._write_procar(tmp_path, _PROCAR_MINIMAL)
        data = _parse_procar_native(p)
        spin = data["spins"][0]
        proj = data["site_proj"][spin]   # (nk=1, nb=2, ni=3)
        # band 1, ion 0 → tot = 0.150
        assert math.isclose(proj[0, 0, 0], 0.150, abs_tol=1e-5)
        # band 1, ion 1 → tot = 0.300
        assert math.isclose(proj[0, 0, 1], 0.300, abs_tol=1e-5)
        # band 2, ion 1 → tot = 0.700
        assert math.isclose(proj[0, 1, 1], 0.700, abs_tol=1e-5)

    def test_eigenvalues(self, tmp_path: Path):
        p    = self._write_procar(tmp_path, _PROCAR_MINIMAL)
        data = _parse_procar_native(p)
        spin = data["spins"][0]
        eigs = data["eigenvalues"][spin]
        assert math.isclose(eigs[0, 0], -2.0, abs_tol=1e-6)
        assert math.isclose(eigs[0, 1],  1.0, abs_tol=1e-6)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _parse_procar_native(tmp_path / "PROCAR_ghost")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Neighbour detection helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestNeighborsFromDefectStructureInfo:

    def _write_dsi(self, path: Path, data: dict) -> None:
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_reads_neighbor_atom_indices_key(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        self._write_dsi(p, {"neighbor_atom_indices": [0, 2, 5]})
        assert neighbors_from_defect_structure_info(p) == [0, 2, 5]

    def test_reads_neighboring_atom_indices_key(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        self._write_dsi(p, {"neighboring_atom_indices": [3, 7]})
        assert neighbors_from_defect_structure_info(p) == [3, 7]

    def test_reads_list_of_dicts_with_index(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        self._write_dsi(p, {"neighbor_atom_indices": [{"index": 1}, {"index": 4}]})
        assert neighbors_from_defect_structure_info(p) == [1, 4]

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        assert neighbors_from_defect_structure_info(tmp_path / "ghost.json") is None

    def test_returns_none_for_unknown_keys(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        self._write_dsi(p, {"something_else": [1, 2]})
        assert neighbors_from_defect_structure_info(p) is None

    def test_returns_none_for_broken_json(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        p.write_text("{broken json")
        assert neighbors_from_defect_structure_info(p) is None

    def test_reads_neighbors_list_of_dicts(self, tmp_path: Path):
        p = tmp_path / "dsi.json"
        self._write_dsi(p, {"neighbors": [{"index": 2, "dist": 1.5}, {"index": 5, "dist": 2.1}]})
        assert neighbors_from_defect_structure_info(p) == [2, 5]


class TestResolveNeighbors:

    def test_prefers_dsi_over_distance(self, tmp_path: Path):
        dsi_path = tmp_path / "dsi.json"
        with open(dsi_path, "w") as fh:
            json.dump({"neighbor_atom_indices": [9, 10]}, fh)

        indices, source = resolve_neighbors(
            dsi_path=dsi_path,
            poscar_path=None,
            defect_center_frac=[0.5, 0.5, 0.5],
        )
        assert indices == [9, 10]
        assert "defect_structure_info" in source

    def test_fallback_to_empty_when_nothing_available(self):
        indices, source = resolve_neighbors(
            dsi_path=None,
            poscar_path=None,
            defect_center_frac=[0.5, 0.5, 0.5],
        )
        assert indices == []
        assert "none" in source.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ParticipationRatioCalculator
# ──────────────────────────────────────────────────────────────────────────────

class TestParticipationRatioCalculator:

    def _make_entry(self, tmp_path: Path) -> Path:
        entry = {"name": "Va_O1_2", "defect_center": [0.25, 0.25, 0.25]}
        p = tmp_path / "defect_entry.json"
        with open(p, "w") as fh:
            json.dump(entry, fh)
        return p

    def _make_dsi(self, tmp_path: Path) -> Path:
        dsi = {"neighbor_atom_indices": list(NEIGHBOR_INDICES)}
        p = tmp_path / "defect_structure_info.json"
        with open(p, "w") as fh:
            json.dump(dsi, fh)
        return p

    def _make_dummy_procar(self, tmp_path: Path) -> Path:
        p = tmp_path / "PROCAR"
        p.write_text("PROCAR dummy\n")
        return p

    def _patch_read_procar(self) -> Any:
        return patch(
            "defectpl.participation_ratio.read_procar",
            return_value=_make_procar_data(),
        )

    def test_run_returns_dict(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc   = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            result = calc.run()

        assert isinstance(result, dict)
        assert "data" in result

    def test_result_property_is_none_before_run(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)
        calc     = ParticipationRatioCalculator(procar=procar_p, defect_entry=entry_p)
        assert calc.result is None

    def test_result_property_set_after_run(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        assert calc.result is not None

    def test_to_json_creates_file(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        out_path = tmp_path / "out.json"
        calc.to_json(out_path)
        assert out_path.exists()
        with open(out_path) as fh:
            loaded = json.load(fh)
        assert loaded["defect_name"] == "Va_O1_2"

    def test_to_json_raises_before_run(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)
        calc     = ParticipationRatioCalculator(procar=procar_p, defect_entry=entry_p)
        with pytest.raises(RuntimeError, match="run()"):
            calc.to_json(tmp_path / "x.json")

    def test_to_csv_creates_file(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        csv_path = tmp_path / "out.csv"
        calc.to_csv(csv_path)
        assert csv_path.exists()

    def test_to_csv_row_count(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        csv_path = tmp_path / "out.csv"
        calc.to_csv(csv_path)
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == N_KPOINTS * N_BANDS

    def test_to_csv_raises_before_run(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)
        calc     = ParticipationRatioCalculator(procar=procar_p, defect_entry=entry_p)
        with pytest.raises(RuntimeError, match="run()"):
            calc.to_csv(tmp_path / "x.csv")

    def test_top_localized_by_p_ratio(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        top = calc.top_localized(n=3, metric="p_ratio")
        assert len(top) == 3
        vals = [r["p_ratio"] for r in top]
        assert vals == sorted(vals, reverse=True)

    def test_top_localized_by_ipr(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            calc.run()

        top  = calc.top_localized(n=5, metric="ipr")
        vals = [r["ipr"] for r in top]
        assert vals == sorted(vals, reverse=True)

    def test_top_localized_raises_before_run(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)
        calc     = ParticipationRatioCalculator(procar=procar_p, defect_entry=entry_p)
        with pytest.raises(RuntimeError):
            calc.top_localized()

    def test_missing_defect_entry_raises(self, tmp_path: Path):
        procar_p = self._make_dummy_procar(tmp_path)
        calc     = ParticipationRatioCalculator(
            procar=procar_p, defect_entry=tmp_path / "nonexistent.json",
        )
        with pytest.raises(FileNotFoundError, match="defect_entry"):
            calc.run()

    def test_neighbor_source_stored_in_result(self, tmp_path: Path):
        entry_p  = self._make_entry(tmp_path)
        dsi_p    = self._make_dsi(tmp_path)
        procar_p = self._make_dummy_procar(tmp_path)

        with self._patch_read_procar():
            calc = ParticipationRatioCalculator(
                procar=procar_p, defect_entry=entry_p, defect_structure_info=dsi_p,
            )
            result = calc.run()

        assert "neighbor_source" in result


# ──────────────────────────────────────────────────────────────────────────────
# 5.  defect_utils — make_defect_entry
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeDefectEntry:

    def test_manual_center(self, tmp_path: Path):
        out = tmp_path / "defect_entry.json"
        payload = make_defect_entry(
            name="Va_O1_2",
            center=[0.5, 0.5, 0.5],
            out_path=out,
        )
        assert out.exists()
        assert payload["name"] == "Va_O1_2"
        assert math.isclose(payload["defect_center"][0], 0.5)

    def test_manual_creates_correct_json(self, tmp_path: Path):
        out = tmp_path / "defect_entry.json"
        make_defect_entry(name="Va_N1_0", center=[0.1, 0.2, 0.3], out_path=out)
        with open(out) as fh:
            data = json.load(fh)
        assert data["name"] == "Va_N1_0"
        assert math.isclose(data["defect_center"][1], 0.2)

    def test_raises_without_center_or_structures(self, tmp_path: Path):
        with pytest.raises(ValueError, match="center"):
            make_defect_entry(name="Va_X", out_path=tmp_path / "x.json")

    def test_output_directory_created(self, tmp_path: Path):
        out = tmp_path / "subdir" / "defect_entry.json"
        make_defect_entry(name="Va_O1_0", center=[0.5, 0.5, 0.5], out_path=out)
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  defect_utils — make_defect_structure_info
# ──────────────────────────────────────────────────────────────────────────────

def _write_simple_poscar(path: Path) -> None:
    """Write a tiny cubic POSCAR with 4 atoms for neighbour tests."""
    poscar = """\
Simple cubic 4 atoms
1.0
4.0  0.0  0.0
0.0  4.0  0.0
0.0  0.0  4.0
N
4
Direct
0.0  0.0  0.0
0.5  0.0  0.0
0.0  0.5  0.0
0.5  0.5  0.0
"""
    path.write_text(poscar)


class TestMakeDefectStructureInfo:

    def test_creates_file(self, tmp_path: Path):
        poscar = tmp_path / "POSCAR"
        _write_simple_poscar(poscar)
        out = tmp_path / "dsi.json"
        make_defect_structure_info(
            poscar=poscar,
            defect_center_frac=[0.0, 0.0, 0.0],
            cutoff_radius=2.5,
            out_path=out,
        )
        assert out.exists()

    def test_json_has_required_keys(self, tmp_path: Path):
        poscar = tmp_path / "POSCAR"
        _write_simple_poscar(poscar)
        out = tmp_path / "dsi.json"
        payload = make_defect_structure_info(
            poscar=poscar,
            defect_center_frac=[0.0, 0.0, 0.0],
            cutoff_radius=2.5,
            out_path=out,
        )
        for key in ("neighbor_atom_indices", "n_neighbors", "cutoff_radius"):
            assert key in payload, f"Missing key: {key}"

    def test_atom_at_center_included(self, tmp_path: Path):
        poscar = tmp_path / "POSCAR"
        _write_simple_poscar(poscar)
        payload = make_defect_structure_info(
            poscar=poscar,
            defect_center_frac=[0.0, 0.0, 0.0],
            cutoff_radius=0.01,
            out_path=tmp_path / "dsi.json",
        )
        # Only atom 0 at (0,0,0) should be within 0.01 Å
        assert 0 in payload["neighbor_atom_indices"]
        assert len(payload["neighbor_atom_indices"]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 7.  defect_utils — parse_frac_coords
# ──────────────────────────────────────────────────────────────────────────────

class TestParseFracCoords:

    def test_comma_separated(self):
        coords = parse_frac_coords("0.5,0.5,0.5")
        assert coords == [0.5, 0.5, 0.5]

    def test_space_separated(self):
        coords = parse_frac_coords("0.25 0.75 0.0")
        assert coords == [0.25, 0.75, 0.0]

    def test_mixed_whitespace_comma(self):
        coords = parse_frac_coords("0.1, 0.2, 0.3")
        assert math.isclose(coords[0], 0.1)

    def test_raises_on_wrong_count(self):
        with pytest.raises(ValueError):
            parse_frac_coords("0.5,0.5")

    def test_raises_on_non_numeric(self):
        with pytest.raises(ValueError):
            parse_frac_coords("a,b,c")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLI tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cli_runner():
    from click.testing import CliRunner
    return CliRunner()


def _setup_calc_dir(tmp_path: Path) -> tuple:
    entry = {"name": "Va_O1_2", "defect_center": [0.5, 0.5, 0.5]}
    (tmp_path / "defect_entry.json").write_text(json.dumps(entry))
    (tmp_path / "defect_structure_info.json").write_text(
        json.dumps({"neighbor_atom_indices": [0, 1]})
    )
    (tmp_path / "PROCAR").write_text("PROCAR placeholder\n")
    return tmp_path, entry


@patch("defectpl.participation_ratio.read_procar", return_value=_make_procar_data())
class TestCLI:

    def test_pr_calc_creates_outputs(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        _setup_calc_dir(tmp_path)
        result = cli_runner.invoke(
            pr_group,
            [
                "calc",
                "--procar", str(tmp_path / "PROCAR"),
                "--entry",  str(tmp_path / "defect_entry.json"),
                "--dsi",    str(tmp_path / "defect_structure_info.json"),
                "--out",    str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "participation_ratio.json").exists()
        assert (tmp_path / "participation_ratio_summary.csv").exists()

    def test_pr_calc_no_csv_flag(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        _setup_calc_dir(tmp_path)
        result = cli_runner.invoke(
            pr_group,
            [
                "calc",
                "--procar", str(tmp_path / "PROCAR"),
                "--entry",  str(tmp_path / "defect_entry.json"),
                "--out",    str(tmp_path),
                "--no-csv",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "participation_ratio_summary.csv").exists()

    def test_pr_summary_reads_json(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        _setup_calc_dir(tmp_path)
        cli_runner.invoke(
            pr_group,
            [
                "calc",
                "--procar", str(tmp_path / "PROCAR"),
                "--entry",  str(tmp_path / "defect_entry.json"),
                "--dsi",    str(tmp_path / "defect_structure_info.json"),
                "--out",    str(tmp_path),
            ],
        )
        json_path = tmp_path / "participation_ratio.json"
        result = cli_runner.invoke(pr_group, ["summary", str(json_path)])
        assert result.exit_code == 0
        assert "Va_O1_2" in result.output

    def test_pr_top_reads_json(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        _setup_calc_dir(tmp_path)
        cli_runner.invoke(
            pr_group,
            [
                "calc",
                "--procar", str(tmp_path / "PROCAR"),
                "--entry",  str(tmp_path / "defect_entry.json"),
                "--dsi",    str(tmp_path / "defect_structure_info.json"),
                "--out",    str(tmp_path),
            ],
        )
        json_path = tmp_path / "participation_ratio.json"
        result = cli_runner.invoke(pr_group, ["top", str(json_path), "--n", "3"])
        assert result.exit_code == 0
        assert "Top 3" in result.output

    def test_pr_batch_processes_subdirs(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        for name in ("Va_O1_0", "Va_O1_1", "Va_O1_2"):
            sub = tmp_path / name
            sub.mkdir()
            _setup_calc_dir(sub)

        result = cli_runner.invoke(
            pr_group,
            ["batch", "--dir", str(tmp_path), "--no-csv"],
        )
        assert result.exit_code == 0, result.output
        for name in ("Va_O1_0", "Va_O1_1", "Va_O1_2"):
            assert (tmp_path / name / "participation_ratio.json").exists()

    def test_pr_batch_writes_combined_csv(self, _mock_rp, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        for name in ("Va_O1_0", "Va_O1_2"):
            sub = tmp_path / name
            sub.mkdir()
            _setup_calc_dir(sub)

        result = cli_runner.invoke(
            pr_group,
            ["batch", "--dir", str(tmp_path), "--combined-csv", "all.csv"],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "all.csv").exists()


class TestCLIMakeEntry:

    def test_make_entry_manual(self, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        out = str(tmp_path / "defect_entry.json")
        result = cli_runner.invoke(
            pr_group,
            ["make-entry", "--name", "Va_O1_2", "--center", "0.5,0.5,0.5", "--out", out],
        )
        assert result.exit_code == 0, result.output
        assert Path(out).exists()
        with open(out) as fh:
            data = json.load(fh)
        assert data["name"] == "Va_O1_2"

    def test_make_entry_no_center_or_structures_fails(self, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        result = cli_runner.invoke(
            pr_group,
            ["make-entry", "--name", "Va_O1_2", "--out", str(tmp_path / "x.json")],
        )
        assert result.exit_code != 0


class TestCLIMakeDsi:

    def test_make_dsi_creates_file(self, cli_runner, tmp_path):
        from defectpl.cli import pr_group

        poscar_path = tmp_path / "POSCAR"
        _write_simple_poscar(poscar_path)
        out = str(tmp_path / "defect_structure_info.json")

        result = cli_runner.invoke(
            pr_group,
            [
                "make-dsi",
                "--poscar", str(poscar_path),
                "--center", "0.0,0.0,0.0",
                "--cutoff", "0.01",
                "--out", out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path(out).exists()