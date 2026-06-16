# -*- coding: utf-8 -*-
"""
Unit tests for the ks_analysis module.
All custom configurations, band edge alignments, and file paths are specified at the top.
"""

import os
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.electronic_structure.core import Spin

# =====================================================================
# Global Test Configuration, Band Edges, & File Paths
# =====================================================================
DATA_DIR = Path(__file__).parent / "data"
REAL_EIGENVAL_PATH: Optional[Path] = DATA_DIR / "pbe/gs" / "EIGENVAL"
OUTPUT_PLOT_DIR: Path = DATA_DIR / "test_outputs"
CUSTOM_STYLE_FILE: Optional[Path] = DATA_DIR / "../../defectpl/defectpl.mplstyle"

# Band edges matching your real physical ground-state EIGENVAL (eV)
VBM_REAL: float = 9.8531
CBM_REAL: float = 13.9744

# Import the module under test
from defectpl.ks_analysis import (
    KohnShamPlotData,
    get_homo_lumo_idx,
    get_spin_multiplicity,
    read_eigenval_file,
    truncate_eigenval,
    find_degenerate_eigenvalues,
    split_energy_occupation,
    xpos_evaluation,
    get_x_values,
    get_occupied_unoccupied_split,
    extract_ksplot_data,
    plot_spin_resolved_levels,
)


class TestKohnShamAnalysis(unittest.TestCase):
    """TestSuite covering Kohn-Sham data modeling, math algorithms, and parsing features."""

    def setUp(self):
        """Pre-populate mock test fixtures representing common VASP eigenvalues outputs."""
        # Clean 1-to-1 mapping representing synthetic [energy, occupancy] bands
        self.mock_up_bands = [
            [-5.0, 1.0],  # 0: Core state
            [-1.2, 1.0],  # 1: VBM state
            [0.1, 1.0],  # 2: Defect level (filled)
            [0.1, 1.0],  # 3: Degenerate defect level (filled)
            [1.5, 0.0],  # 4: Defect level (empty)
            [3.0, 0.0],  # 5: CBM state
        ]
        self.mock_down_bands = [
            [-5.0, 1.0],
            [-1.4, 1.0],
            [0.15, 0.0],  # Unpaired spin-down defect counterstate (empty)
            [0.15, 0.0],
            [1.6, 0.0],
            [3.2, 0.0],
        ]

        # Ensure output directory exists for plot dumps
        OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def test_kohn_sham_plot_data_serialization(self):
        """Verifies that KohnShamPlotData properly satisfies MSONable dict round-trips."""
        data_instance = KohnShamPlotData(
            up=self.mock_up_bands[2:4],
            down=self.mock_down_bands[2:4],
            up_idx=[2, 3],
            down_idx=[2, 3],
            up_energies=[0.1, 0.1],
            up_occupations=[1.0, 1.0],
            down_energies=[0.15, 0.15],
            down_occupations=[0.0, 0.0],
            degenerate_up=[[0, 1]],
            degenerate_down=[[0, 1]],
            max_div_up=2,
            max_div_down=2,
            xvalues_up=[-4.5, -5.5],
            xvalues_down=[4.5, 5.5],
            occupied_up={"xvalues": [-4.5, -5.5], "energies": [0.1, 0.1]},
            unoccupied_up={"xvalues": [], "energies": []},
            occupied_down={"xvalues": [], "energies": []},
            unoccupied_down={"xvalues": [4.5, 5.5], "energies": [0.15, 0.15]},
            vbm=-1.2,
            cbm=3.0,
            emin=-2.2,
            emax=4.0,
            espan=1.0,
            sep=0.1,
            lim=10.0,
            w=4.55,
            meta_info={"spin_multiplicity": 3.0},
        )

        serialized_dict = data_instance.as_dict()
        self.assertEqual(serialized_dict["@class"], "KohnShamPlotData")

        reconstructed_instance = KohnShamPlotData.from_dict(serialized_dict)
        self.assertEqual(reconstructed_instance.vbm, data_instance.vbm)
        self.assertEqual(reconstructed_instance.cbm, data_instance.cbm)
        self.assertListEqual(
            reconstructed_instance.up_energies, data_instance.up_energies
        )

    def test_get_homo_lumo_idx(self):
        """Validates correct index boundaries resolution marking HOMO/LUMO separation."""
        homo_up, lumo_up = get_homo_lumo_idx(self.mock_up_bands, thr=0.6)
        self.assertEqual(homo_up, 3)
        self.assertEqual(lumo_up, 4)

        fractional_bands = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        with self.assertWarns(UserWarning):
            get_homo_lumo_idx(fractional_bands, thr=0.6)

    def test_get_spin_multiplicity(self):
        """Asserts correct theoretical equation mapping logic tracking 2S + 1 properties."""
        self.assertEqual(get_spin_multiplicity(homo_up_idx=3, homo_down_idx=3), 1.0)
        self.assertEqual(get_spin_multiplicity(homo_up_idx=4, homo_down_idx=3), 2.0)
        self.assertEqual(get_spin_multiplicity(homo_up_idx=5, homo_down_idx=3), 3.0)

    @patch("defectpl.vasp.Eigenval")
    def test_read_eigenval_file_mocked(self, mock_eigenval_class):
        """Verifies parsing routing and key injection structures using mock inputs."""
        mock_instance = MagicMock()
        mock_instance.ispin = 2
        mock_instance.nkpt = 1
        mock_instance.nelect = 12.0
        mock_instance.nbands = 6
        mock_instance.kpoints = [[0.0, 0.0, 0.0]]

        mock_instance.eigenvalue_band_properties = (
            [4.2, 4.6],
            [3.0, 3.2],
            [-1.2, -1.4],
        )

        mock_instance.eigenvalues = {
            Spin.up: [np.array(self.mock_up_bands)],
            Spin.down: [np.array(self.mock_down_bands)],
        }
        mock_eigenval_class.return_value = mock_instance

        parsed_data = read_eigenval_file("dummy_path/EIGENVAL", k_idx=0)

        self.assertEqual(parsed_data["nelect"], 12.0)
        self.assertEqual(parsed_data["homo_up_idx"], 3)
        self.assertEqual(parsed_data["homo_down_idx"], 1)
        self.assertEqual(parsed_data["spin_multiplicity"], 3.0)
        self.assertEqual(parsed_data["hl_gap_up"], 4.2)

    def test_read_eigenval_file_non_spin_polarized_raises(self):
        """Confirms system abort instances throw errors if calculations are not spin-polarized."""
        with patch("defectpl.vasp.Eigenval") as mock_eigenval_class:
            mock_instance = MagicMock()
            mock_instance.ispin = 1
            mock_eigenval_class.return_value = mock_instance

            with self.assertRaises(ValueError):
                read_eigenval_file("dummy_path/EIGENVAL", k_idx=0)

    def test_truncate_eigenval(self):
        """Validates that energy window clipping and index tracking operate accurately."""
        trunc, indices = truncate_eigenval(self.mock_up_bands, emin=-2.0, emax=2.0)
        self.assertEqual(len(trunc), 4)
        self.assertListEqual(indices, [1, 2, 3, 4])
        self.assertEqual(trunc[0][0], -1.2)
        self.assertEqual(trunc[-1][0], 1.5)

    def test_find_degenerate_eigenvalues(self):
        """Verifies clustering groupings tracking near-identical eigenvalue arrays elements."""
        degenerate_matrix = [
            [0.100, 1.0],
            [0.101, 1.0],
            [0.500, 0.0],
            [0.500, 0.0],
        ]
        groups = find_degenerate_eigenvalues(degenerate_matrix, tol=1e-2)
        self.assertListEqual(groups, [[0, 1], [2, 3]])

    def test_split_energy_occupation(self):
        """Confirms clean splitting mechanics filtering paired parameters down into standalone arrays."""
        energies, occupations = split_energy_occupation(self.mock_up_bands)
        self.assertEqual(len(energies), len(self.mock_up_bands))
        self.assertEqual(energies[0], -5.0)
        self.assertEqual(occupations[0], 1.0)

    def test_xpos_evaluation(self):
        """Tests that spatial offset computations map coordinate lines systematically on the canvas."""
        self.assertListEqual(xpos_evaluation(npoint=1, max_div=1, lim=10.0), [5.0])
        self.assertListEqual(xpos_evaluation(npoint=0, max_div=1, lim=10.0), [0.0])

        even_pos = xpos_evaluation(npoint=2, max_div=2, sep=1.0, lim=10.0)
        self.assertEqual(len(even_pos), 2)
        self.assertIn(7.25, even_pos)
        self.assertIn(2.75, even_pos)

    def test_get_occupied_unoccupied_split(self):
        """Validates clear structural partition sorting variables across occupancy definitions."""
        occupations = [1.0, 0.0, 1.0]
        xvalues = [-1.0, -2.0, -3.0]
        energies = [0.1, 0.2, 0.3]

        occ, unocc = get_occupied_unoccupied_split(
            occupations, xvalues, energies, threshold=0.6
        )
        self.assertListEqual(occ["energies"], [0.1, 0.3])
        self.assertListEqual(occ["xvalues"], [-1.0, -3.0])
        self.assertListEqual(unocc["energies"], [0.2])

    def test_extract_ksplot_data(self):
        """Performs integrated flow tracing checking end-to-end data processing functions pipelines."""
        eigenval_data = {
            "up": self.mock_up_bands,
            "down": self.mock_down_bands,
            "selected_kpoint": [0, [0.0, 0.0, 0.0]],
            "spin_multiplicity": 3.0,
            "nelect": 12.0,
        }

        plot_data = extract_ksplot_data(eigenval_data, vbm=-1.2, cbm=3.0, espan=1.0)

        self.assertIsInstance(plot_data, KohnShamPlotData)
        self.assertEqual(plot_data.vbm, -1.2)
        self.assertEqual(plot_data.cbm, 3.0)
        self.assertEqual(plot_data.emin, -2.2)
        self.assertEqual(plot_data.emax, 4.0)
        self.assertTrue(all(x < 0 for x in plot_data.xvalues_up))
        self.assertTrue(all(x > 0 for x in plot_data.xvalues_down))

    def test_plot_spin_resolved_levels_execution(self):
        """Confirms visualization routine handles array outputs and writes files correctly."""
        eigenval_data = {
            "up": self.mock_up_bands,
            "down": self.mock_down_bands,
            "selected_kpoint": [0, [0.0, 0.0, 0.0]],
            "spin_multiplicity": 3.0,
            "nelect": 12.0,
        }
        plot_data = extract_ksplot_data(eigenval_data, vbm=-1.2, cbm=3.0, espan=1.0)

        test_img_path = OUTPUT_PLOT_DIR / "test_ks_output.png"

        try:
            plot_spin_resolved_levels(
                data=plot_data,
                output_filename=test_img_path,
                style_file=(
                    str(CUSTOM_STYLE_FILE)
                    if (CUSTOM_STYLE_FILE and CUSTOM_STYLE_FILE.exists())
                    else None
                ),
            )
            self.assertTrue(test_img_path.exists())
        finally:
            if test_img_path.exists():
                test_img_path.unlink()

    # =====================================================================
    # Live Data Integration & Real Level Parsing Checkpoint
    # =====================================================================
    def test_live_eigenval_file_parsing(self):
        """
        Runs complete extraction pipeline validations using real EIGENVAL
        files and the statically defined global band edge thresholds.
        """
        if REAL_EIGENVAL_PATH is None or not REAL_EIGENVAL_PATH.exists():
            self.skipTest(
                f"Target EIGENVAL file missing at {REAL_EIGENVAL_PATH}. Skipping live file integration tests."
            )

        print(f"\nExecuting Live Integration Analysis against: {REAL_EIGENVAL_PATH}")

        # 1. Parse raw k-point eigenvalues dictionary payload from filesystem
        live_raw_data = read_eigenval_file(REAL_EIGENVAL_PATH, k_idx=0)
        self.assertIn("up", live_raw_data)
        self.assertIn("down", live_raw_data)
        self.assertGreaterEqual(live_raw_data["nbands"], 1)

        # 2. Extract layout dimensions using top-level configured VBM/CBM alignments
        processed_plot_data = extract_ksplot_data(
            eigenval_data=live_raw_data, vbm=VBM_REAL, cbm=CBM_REAL, espan=1.5
        )

        # 3. Assert correct bounds enforcement matching our real PBE input constraints
        self.assertEqual(processed_plot_data.vbm, VBM_REAL)
        self.assertEqual(processed_plot_data.cbm, CBM_REAL)
        self.assertAlmostEqual(processed_plot_data.emin, VBM_REAL - 1.5)
        self.assertAlmostEqual(processed_plot_data.emax, CBM_REAL + 1.5)

        # 4. Attempt rendering live parsed records to disk
        live_img_path = OUTPUT_PLOT_DIR / "live_ks_pbe_plot.png"
        try:
            plot_spin_resolved_levels(
                data=processed_plot_data,
                output_filename=live_img_path,
                style_file=(
                    str(CUSTOM_STYLE_FILE)
                    if (CUSTOM_STYLE_FILE and CUSTOM_STYLE_FILE.exists())
                    else None
                ),
            )
            self.assertTrue(live_img_path.exists())
            print(
                f"Generated physical validation plot successfully at: {live_img_path}"
            )
        finally:
            if live_img_path.exists():
                live_img_path.unlink()


if __name__ == "__main__":
    unittest.main()
