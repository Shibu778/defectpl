# -*- coding: utf-8 -*-
"""
Unit tests for participation-ratio plotting functions and the associated
CLI subcommands  ``defectpl pr plot``  and  ``defectpl pr ksplot``.

All tests run headless (matplotlib Agg backend) and do not require any
real VASP output files.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

# ── imports under test ────────────────────────────────────────────────────────
from defectpl.participation_ratio import (
    flatten_pr_result,
    plot_pr_vs_energy,
    plot_pr_vs_band_index,
    _METRIC_LABEL,
    _SPIN_COLORS,
)
from defectpl.ks_analysis import (
    KohnShamPlotData,
    plot_ks_with_pr,
    _lookup_pr_values,
)
from defectpl.cli import pr_group


# ═════════════════════════════════════════════════════════════════════════════
# Shared synthetic fixtures
# ═════════════════════════════════════════════════════════════════════════════

def _make_pr_result(n_bands: int = 10, n_spins: int = 1, n_kpts: int = 1) -> dict:
    """
    Build a minimal synthetic participation_ratio.json-compatible dict.

    Bands are numbered 1..n_bands.  Energy ramps linearly from -1 to +1 eV.
    P-ratio ramps from 0 to 1; IPR is set to 1/n_bands for all bands.
    Occupation is 1.0 for the lower half, 0.0 for the upper half.
    """
    rng   = np.random.default_rng(42)
    spins = ["spin_1"] if n_spins == 1 else ["spin_1", "spin_2"]
    data: dict = {}

    for sp in spins:
        data[sp] = {}
        for ik in range(n_kpts):
            kpt_label = f"kpt_{ik + 1}"
            data[sp][kpt_label] = {}
            for ib in range(n_bands):
                energy  = -1.0 + 2.0 * ib / max(n_bands - 1, 1)
                occ     = 1.0 if ib < n_bands // 2 else 0.0
                p_ratio = float(rng.uniform(0.0, 1.0))
                ipr     = float(1.0 / n_bands)
                data[sp][kpt_label][f"band_{ib + 1}"] = {
                    "energy":      energy,
                    "occupation":  occ,
                    "p_ratio":     p_ratio,
                    "ipr":         ipr,
                    "p_neighbors": p_ratio * 0.9,
                    "p_total":     1.0,
                }

    return {
        "defect_name":           "Va_O1",
        "defect_center":         [0.5, 0.5, 0.5],
        "neighbor_atom_indices": [0, 1, 2, 3],
        "n_atoms":               32,
        "n_spins":               n_spins,
        "n_kpoints":             n_kpts,
        "n_bands":               n_bands,
        "data":                  data,
    }


def _make_ks_data(n_up: int = 6, n_down: int = 6,
                  vbm: float = -0.5, cbm: float = 0.5) -> KohnShamPlotData:
    """
    Build a minimal KohnShamPlotData container with linear energy levels.
    """
    def _make_eigenvals(n, e_lo=-1.5, e_hi=1.5):
        return [[e_lo + (e_hi - e_lo) * i / max(n - 1, 1), 1.0 if i < n // 2 else 0.0]
                for i in range(n)]

    up   = _make_eigenvals(n_up)
    down = _make_eigenvals(n_down)

    def _split(eig):
        return [e for e, _ in eig], [o for _, o in eig]

    up_e,   up_o   = _split(up)
    down_e, down_o = _split(down)

    xup   = [-5.0 + i * (10.0 / max(n_up - 1, 1))   for i in range(n_up)]
    xdown = [-5.0 + i * (10.0 / max(n_down - 1, 1)) for i in range(n_down)]

    emin = vbm - 1.0
    emax = cbm + 1.0
    lim  = 10.0

    def _occ_split(xvals, energies, occs):
        occ  = {"xvalues": [x for x, o in zip(xvals, occs) if o > 0.6],
                "energies": [e for e, o in zip(energies, occs) if o > 0.6]}
        emp  = {"xvalues": [x for x, o in zip(xvals, occs) if o <= 0.6],
                "energies": [e for e, o in zip(energies, occs) if o <= 0.6]}
        return occ, emp

    occ_up,   emp_up   = _occ_split(xup,   up_e,   up_o)
    occ_down, emp_down = _occ_split(xdown, down_e, down_o)

    return KohnShamPlotData(
        up=up,
        down=down,
        up_idx=list(range(n_up)),
        down_idx=list(range(n_down)),
        up_energies=up_e,
        up_occupations=up_o,
        down_energies=down_e,
        down_occupations=down_o,
        degenerate_up=[[i] for i in range(n_up)],
        degenerate_down=[[i] for i in range(n_down)],
        max_div_up=1,
        max_div_down=1,
        xvalues_up=xup,
        xvalues_down=xdown,
        occupied_up=occ_up,
        unoccupied_up=emp_up,
        occupied_down=occ_down,
        unoccupied_down=emp_down,
        vbm=vbm,
        cbm=cbm,
        emin=emin,
        emax=emax,
        espan=1.0,
        sep=0.1,
        lim=lim,
        w=1.0,
        meta_info={},
    )


# ═════════════════════════════════════════════════════════════════════════════
# flatten_pr_result
# ═════════════════════════════════════════════════════════════════════════════

class TestFlattenPrResult:

    def test_returns_list_of_dicts(self):
        result = _make_pr_result(n_bands=5)
        rows = flatten_pr_result(result)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_correct_row_count_single_spin(self):
        result = _make_pr_result(n_bands=8, n_spins=1)
        rows = flatten_pr_result(result, kpt_idx=0)
        assert len(rows) == 8

    def test_correct_row_count_two_spins(self):
        result = _make_pr_result(n_bands=8, n_spins=2)
        rows = flatten_pr_result(result, kpt_idx=0)
        assert len(rows) == 16

    def test_required_keys_present(self):
        result = _make_pr_result(n_bands=4)
        row = flatten_pr_result(result)[0]
        for key in ("spin", "band", "energy", "occ", "p_ratio", "ipr"):
            assert key in row, f"Key '{key}' missing from flattened row"

    def test_band_indices_are_integers(self):
        result = _make_pr_result(n_bands=6)
        rows = flatten_pr_result(result)
        for r in rows:
            assert isinstance(r["band"], int)

    def test_kpt_idx_filtering(self):
        result = _make_pr_result(n_bands=4, n_kpts=2)
        rows0 = flatten_pr_result(result, kpt_idx=0)
        rows1 = flatten_pr_result(result, kpt_idx=1)
        assert len(rows0) == 4
        assert len(rows1) == 4

    def test_missing_kpt_returns_empty(self):
        result = _make_pr_result(n_bands=4, n_kpts=1)
        rows = flatten_pr_result(result, kpt_idx=99)
        assert rows == []

    def test_p_ratio_in_range(self):
        result = _make_pr_result(n_bands=10)
        for r in flatten_pr_result(result):
            assert 0.0 <= r["p_ratio"] <= 1.0

    def test_occ_coerced_to_float(self):
        result = _make_pr_result(n_bands=4)
        for r in flatten_pr_result(result):
            assert isinstance(r["occ"], float)


# ═════════════════════════════════════════════════════════════════════════════
# plot_pr_vs_energy
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotPrVsEnergy:

    def setup_method(self):
        self.result = _make_pr_result(n_bands=10, n_spins=2)

    def test_returns_axes(self):
        ax = plot_pr_vs_energy(self.result)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_y_label_p_ratio(self):
        ax = plot_pr_vs_energy(self.result, metric="p_ratio")
        assert "P-ratio" in ax.get_ylabel() or "p_ratio" in ax.get_ylabel().lower()
        plt.close("all")

    def test_y_label_ipr(self):
        ax = plot_pr_vs_energy(self.result, metric="ipr")
        assert "IPR" in ax.get_ylabel() or "ipr" in ax.get_ylabel().lower()
        plt.close("all")

    def test_title_defaults_to_defect_name(self):
        ax = plot_pr_vs_energy(self.result)
        assert "Va_O1" in ax.get_title()
        plt.close("all")

    def test_custom_title(self):
        ax = plot_pr_vs_energy(self.result, title="My Title")
        assert ax.get_title() == "My Title"
        plt.close("all")

    def test_saves_file(self, tmp_path):
        out = tmp_path / "test_energy.png"
        plot_pr_vs_energy(self.result, out=out)
        assert out.exists()
        plt.close("all")

    def test_energy_filter_emin(self):
        ax = plot_pr_vs_energy(self.result, emin=0.0)
        # All scatter points should have x >= 0
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets):
                xs = offsets[:, 0]
                assert (xs >= 0.0 - 1e-6).all(), "Point below emin found"
        plt.close("all")

    def test_energy_filter_emax(self):
        ax = plot_pr_vs_energy(self.result, emax=0.0)
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets):
                xs = offsets[:, 0]
                assert (xs <= 0.0 + 1e-6).all(), "Point above emax found"
        plt.close("all")

    def test_threshold_line_present(self):
        ax = plot_pr_vs_energy(self.result, threshold=0.3)
        hlines = [l for l in ax.lines if l.get_linestyle() == "--"]
        assert len(hlines) >= 1, "No dashed threshold line found"
        plt.close("all")

    def test_vbm_cbm_vertical_lines(self):
        ax = plot_pr_vs_energy(self.result, vbm=-0.5, cbm=0.5)
        vlines = [l for l in ax.lines if l.get_linestyle() == ":"]
        assert len(vlines) >= 2, "Expected at least 2 vertical lines (VBM, CBM)"
        plt.close("all")

    def test_inject_axes(self):
        fig, ax_in = plt.subplots()
        ax_out = plot_pr_vs_energy(self.result, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_single_spin(self):
        result = _make_pr_result(n_bands=6, n_spins=1)
        ax = plot_pr_vs_energy(result)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# ═════════════════════════════════════════════════════════════════════════════
# plot_pr_vs_band_index
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotPrVsBandIndex:

    def setup_method(self):
        self.result = _make_pr_result(n_bands=12, n_spins=2)

    def test_returns_axes(self):
        ax = plot_pr_vs_band_index(self.result)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_x_label_band_index(self):
        ax = plot_pr_vs_band_index(self.result)
        assert "band" in ax.get_xlabel().lower() or "Band" in ax.get_xlabel()
        plt.close("all")

    def test_y_label_p_ratio(self):
        ax = plot_pr_vs_band_index(self.result, metric="p_ratio")
        assert "P-ratio" in ax.get_ylabel() or "ratio" in ax.get_ylabel().lower()
        plt.close("all")

    def test_y_label_ipr(self):
        ax = plot_pr_vs_band_index(self.result, metric="ipr")
        assert "IPR" in ax.get_ylabel() or "ipr" in ax.get_ylabel().lower()
        plt.close("all")

    def test_title_defaults_to_defect_name(self):
        ax = plot_pr_vs_band_index(self.result)
        assert "Va_O1" in ax.get_title()
        plt.close("all")

    def test_saves_file(self, tmp_path):
        out = tmp_path / "test_band.png"
        plot_pr_vs_band_index(self.result, out=out)
        assert out.exists()
        plt.close("all")

    def test_energy_window_filter(self):
        # Only bands with energy in [0, 1] should appear
        ax = plot_pr_vs_band_index(self.result, emin=0.0, emax=1.0)
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets):
                # x-axis is band index (integer), check they're positive
                xs = offsets[:, 0]
                assert (xs >= 1).all()
        plt.close("all")

    def test_threshold_line_present(self):
        ax = plot_pr_vs_band_index(self.result, threshold=0.15)
        hlines = [l for l in ax.lines if l.get_linestyle() == "--"]
        assert len(hlines) >= 1
        plt.close("all")

    def test_inject_axes(self):
        fig, ax_in = plt.subplots()
        ax_out = plot_pr_vs_band_index(self.result, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_scatter_count_matches_bands(self):
        result = _make_pr_result(n_bands=8, n_spins=1)
        ax = plot_pr_vs_band_index(result)
        total_pts = sum(len(c.get_offsets()) for c in ax.collections)
        assert total_pts == 8
        plt.close("all")


# ═════════════════════════════════════════════════════════════════════════════
# _lookup_pr_values
# ═════════════════════════════════════════════════════════════════════════════

class TestLookupPrValues:

    def setup_method(self):
        self.result = _make_pr_result(n_bands=10, n_spins=2)

    def test_correct_length(self):
        vals = _lookup_pr_values(self.result, [0, 2, 4], "spin_1", "p_ratio")
        assert len(vals) == 3

    def test_returns_float(self):
        vals = _lookup_pr_values(self.result, [0], "spin_1", "p_ratio")
        assert isinstance(vals[0], float)

    def test_missing_band_gives_nan(self):
        vals = _lookup_pr_values(self.result, [999], "spin_1", "p_ratio")
        assert vals[0] != vals[0]  # NaN

    def test_ipr_values_in_range(self):
        n = self.result["n_bands"]
        vals = _lookup_pr_values(self.result, list(range(n)), "spin_1", "ipr")
        for v in vals:
            assert 0.0 <= v <= 1.0

    def test_spin2_lookup(self):
        vals = _lookup_pr_values(self.result, [0, 1], "spin_2", "p_ratio")
        assert len(vals) == 2
        assert all(v == v for v in vals)  # not NaN

    def test_wrong_spin_gives_nan(self):
        vals = _lookup_pr_values(self.result, [0], "spin_99", "p_ratio")
        assert vals[0] != vals[0]  # NaN


# ═════════════════════════════════════════════════════════════════════════════
# plot_ks_with_pr
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotKsWithPr:

    def setup_method(self):
        self.ks_data  = _make_ks_data(n_up=6, n_down=6)
        self.pr_result = _make_pr_result(n_bands=6, n_spins=2)

    def test_creates_output_file(self, tmp_path):
        out = tmp_path / "ks_pr.png"
        plot_ks_with_pr(self.ks_data, self.pr_result, output_filename=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")

    def test_creates_pdf(self, tmp_path):
        out = tmp_path / "ks_pr.pdf"
        plot_ks_with_pr(self.ks_data, self.pr_result, output_filename=out)
        assert out.exists()
        plt.close("all")

    def test_single_spin_fallback(self, tmp_path):
        out = tmp_path / "ks_pr_single.png"
        pr1 = _make_pr_result(n_bands=6, n_spins=1)
        plot_ks_with_pr(self.ks_data, pr1, output_filename=out)
        assert out.exists()
        plt.close("all")

    def test_ipr_metric(self, tmp_path):
        out = tmp_path / "ks_ipr.png"
        plot_ks_with_pr(self.ks_data, self.pr_result, metric="ipr",
                        output_filename=out)
        assert out.exists()
        plt.close("all")

    def test_custom_cmap(self, tmp_path):
        out = tmp_path / "ks_cmap.png"
        plot_ks_with_pr(self.ks_data, self.pr_result, cmap="viridis",
                        output_filename=out)
        assert out.exists()
        plt.close("all")

    def test_custom_vmin_vmax(self, tmp_path):
        out = tmp_path / "ks_vminvmax.png"
        plot_ks_with_pr(self.ks_data, self.pr_result, vmin=0.1, vmax=0.5,
                        output_filename=out)
        assert out.exists()
        plt.close("all")

    def test_title_set_from_defect_name(self, tmp_path):
        out = tmp_path / "ks_title.png"
        plot_ks_with_pr(self.ks_data, self.pr_result, output_filename=out)
        plt.close("all")

    def test_more_bands_in_pr_than_ks(self, tmp_path):
        out = tmp_path / "ks_extra.png"
        pr_big = _make_pr_result(n_bands=20, n_spins=2)
        plot_ks_with_pr(self.ks_data, pr_big, output_filename=out)
        assert out.exists()
        plt.close("all")


# ═════════════════════════════════════════════════════════════════════════════
# CLI — defectpl pr plot (energy mode)
# ═════════════════════════════════════════════════════════════════════════════

class TestCliPrPlotEnergy:

    def setup_method(self):
        self.runner = CliRunner()
        self.result = _make_pr_result(n_bands=10, n_spins=2)

    def _invoke(self, args, result_dict=None):
        if result_dict is None:
            result_dict = self.result
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "participation_ratio.json")
            out_path  = os.path.join(td, "out.png")
            with open(json_path, "w") as fh:
                json.dump(result_dict, fh)
            rv = self.runner.invoke(pr_group, ["plot", json_path,
                                               "--out", out_path] + args)
            return rv, out_path

    def test_energy_mode_exit_zero(self):
        rv, _ = self._invoke(["--xaxis", "energy"])
        assert rv.exit_code == 0, rv.output

    def test_energy_mode_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "pr.json")
            out_path  = os.path.join(td, "pr_energy.png")
            with open(json_path, "w") as fh:
                json.dump(self.result, fh)
            rv = self.runner.invoke(pr_group, [
                "plot", json_path, "--xaxis", "energy", "--out", out_path
            ])
            assert rv.exit_code == 0
            assert os.path.exists(out_path)

    def test_ipr_metric(self):
        rv, _ = self._invoke(["--xaxis", "energy", "--metric", "ipr"])
        assert rv.exit_code == 0

    def test_threshold_option(self):
        rv, _ = self._invoke(["--xaxis", "energy", "--threshold", "0.15"])
        assert rv.exit_code == 0

    def test_vbm_cbm_options(self):
        rv, _ = self._invoke(["--xaxis", "energy", "--vbm", "-0.5", "--cbm", "0.5"])
        assert rv.exit_code == 0

    def test_emin_emax_options(self):
        rv, _ = self._invoke(["--xaxis", "energy", "--emin", "-0.5", "--emax", "0.5"])
        assert rv.exit_code == 0


# ═════════════════════════════════════════════════════════════════════════════
# CLI — defectpl pr plot (band mode)
# ═════════════════════════════════════════════════════════════════════════════

class TestCliPrPlotBand:

    def setup_method(self):
        self.runner = CliRunner()
        self.result = _make_pr_result(n_bands=10, n_spins=2)

    def test_band_mode_exit_zero(self):
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "pr.json")
            out_path  = os.path.join(td, "pr_band.png")
            with open(json_path, "w") as fh:
                json.dump(self.result, fh)
            rv = self.runner.invoke(pr_group, [
                "plot", json_path, "--xaxis", "band", "--out", out_path
            ])
            assert rv.exit_code == 0, rv.output
            assert os.path.exists(out_path)

    def test_band_mode_with_energy_filter(self):
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "pr.json")
            out_path  = os.path.join(td, "pr_band_filt.png")
            with open(json_path, "w") as fh:
                json.dump(self.result, fh)
            rv = self.runner.invoke(pr_group, [
                "plot", json_path, "--xaxis", "band",
                "--emin", "-0.5", "--emax", "0.5",
                "--out", out_path,
            ])
            assert rv.exit_code == 0

    def test_default_xaxis_is_energy(self):
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "pr.json")
            out_path  = os.path.join(td, "default.png")
            with open(json_path, "w") as fh:
                json.dump(self.result, fh)
            rv = self.runner.invoke(pr_group, [
                "plot", json_path, "--out", out_path,
            ])
            assert rv.exit_code == 0


# ═════════════════════════════════════════════════════════════════════════════
# CLI — defectpl pr ksplot
# ═════════════════════════════════════════════════════════════════════════════

class TestCliPrKsplot:

    def setup_method(self):
        self.runner = CliRunner()
        self.pr_result = _make_pr_result(n_bands=6, n_spins=2)
        self.ks_data   = _make_ks_data(n_up=6, n_down=6)

    def _invoke_ksplot(self, extra_args=None):
        with tempfile.TemporaryDirectory() as td:
            pr_path  = os.path.join(td, "participation_ratio.json")
            out_path = os.path.join(td, "ks_pr.png")
            with open(pr_path, "w") as fh:
                json.dump(self.pr_result, fh)

            # Patch at the source module (CLI uses lazy local imports inside the function)
            with patch("defectpl.vasp.read_eigenval_file") as mock_re, \
                 patch("defectpl.ks_analysis.extract_ksplot_data") as mock_ek, \
                 patch("defectpl.ks_analysis.plot_ks_with_pr") as mock_pk:

                mock_re.return_value  = {"up": [], "down": [], "nelect": 6.0,
                                          "homo_up_idx": 2, "homo_down_idx": 2,
                                          "spin_multiplicity": 1.0, "hl_gap_up": 1.0,
                                          "selected_kpoint": [0, 0, 0]}
                mock_ek.return_value  = self.ks_data
                mock_pk.return_value  = None

                # Create a dummy EIGENVAL so click's exists=True check passes
                eigenval_path = os.path.join(td, "EIGENVAL")
                Path(eigenval_path).write_text("dummy")

                args = [
                    "ksplot",
                    "--eigenval", eigenval_path,
                    "--pr-json", pr_path,
                    "--vbm", "-0.5",
                    "--cbm",  "0.5",
                    "--out", out_path,
                ] + (extra_args or [])
                rv = self.runner.invoke(pr_group, args)
                return rv, mock_pk

    def test_exit_code_zero(self):
        rv, _ = self._invoke_ksplot()
        assert rv.exit_code == 0, rv.output

    def test_plot_function_called(self):
        _, mock_pk = self._invoke_ksplot()
        mock_pk.assert_called_once()

    def test_metric_passed_through(self):
        _, mock_pk = self._invoke_ksplot(["--metric", "ipr"])
        _, kwargs = mock_pk.call_args
        assert kwargs.get("metric") == "ipr"

    def test_cmap_passed_through(self):
        _, mock_pk = self._invoke_ksplot(["--cmap", "viridis"])
        _, kwargs = mock_pk.call_args
        assert kwargs.get("cmap") == "viridis"

    def test_vmin_vmax_passed(self):
        _, mock_pk = self._invoke_ksplot(["--vmin", "0.1", "--vmax", "0.8"])
        _, kwargs = mock_pk.call_args
        assert abs(kwargs.get("vmin") - 0.1) < 1e-6
        assert abs(kwargs.get("vmax") - 0.8) < 1e-6

    def test_missing_vbm_fails(self):
        with tempfile.TemporaryDirectory() as td:
            pr_path = os.path.join(td, "pr.json")
            with open(pr_path, "w") as fh:
                json.dump(self.pr_result, fh)
            eigenval_path = os.path.join(td, "EIGENVAL")
            Path(eigenval_path).write_text("dummy")
            rv = self.runner.invoke(pr_group, [
                "ksplot",
                "--eigenval", eigenval_path,
                "--pr-json",  pr_path,
                "--cbm", "0.5",   # --vbm missing
            ])
            assert rv.exit_code != 0


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_flatten_empty_result(self):
        result = {"defect_name": "test", "data": {}}
        rows = flatten_pr_result(result)
        assert rows == []

    def test_plot_energy_empty_after_filter(self):
        result = _make_pr_result(n_bands=4)
        ax = plot_pr_vs_energy(result, emin=100.0, emax=200.0)
        # Should still return axes even with no data after filter
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_band_handles_none_energy(self):
        result = _make_pr_result(n_bands=4)
        result["data"]["spin_1"]["kpt_1"]["band_2"]["energy"] = None
        ax = plot_pr_vs_band_index(result)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_pr_vs_energy_no_out_does_not_save(self, tmp_path):
        result = _make_pr_result(n_bands=4)
        ax = plot_pr_vs_energy(result)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_lookup_empty_indices(self):
        result = _make_pr_result(n_bands=5)
        vals = _lookup_pr_values(result, [], "spin_1", "p_ratio")
        assert vals == []
