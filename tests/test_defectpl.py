# -*- coding: utf-8 -*-
"""
Unit tests for the core defectpl module containing Photoluminescence,
VibrationalSpectra1D, and ConfigurationCoordinateDiagram data classes.

All global system parameters and path definitions are declared at the top.
"""

import json
import os
import unittest
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from monty.json import MSONable
from pymatgen.core import Lattice, Structure

# =====================================================================
# Global Test Configuration & File Paths
# =====================================================================
OUTPUT_TEST_DIR: Path = Path("./defectpl_test_outputs")
MPL_STYLE_FILE: Optional[Path] = Path("../defectpl/defectpl.mplstyle")

# Configuration Coordinate setup directories
MOCK_INPUT_DIR_G: Path = Path("data/pbe/gs/")
MOCK_INPUT_DIR_E: Path = Path("data/pbe/zpl")

# Import the module under test
from defectpl.defectpl import (
    Photoluminescence,
    VibrationalSpectra1D,
    ConfigurationCoordinateDiagram,
)


class TestPhotoluminescence(unittest.TestCase):
    """TestSuite covering the multi-mode functional Photoluminescence engine."""

    def setUp(self):
        """Initialize array dimensions matching standard points configurations."""
        self.frequencies = np.array([0.02, 0.04, 0.06], dtype=float)  # eV
        # shape: (nmodes, natoms, 3) -> 3 modes, 2 atoms, 3 dimensions
        self.eigenvectors = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
        ], dtype=float)
        self.masses = np.array([12.011, 14.007], dtype=float)        # C and N in AMU
        self.dR = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]], dtype=float)  # Å
        
        self.EZPL = 2.0
        self.gamma = 0.005

        OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    @patch("defectpl.defectpl.Plotter")
    def test_core_pipeline_and_serialization(self, mock_plotter_class):
        """Validates that computed physical properties and MSONable round-trips align perfectly."""
        # 1. Instantiate Core Engine
        pl_engine = Photoluminescence(
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            masses=self.masses,
            dR=self.dR,
            EZPL=self.EZPL,
            gamma=self.gamma,
            resolution=100,
            max_energy=0.5
        )

        # 2. Check automated physics processing hooks
        self.assertIsNotNone(pl_engine.HR_factor)
        self.assertIsNotNone(pl_engine.intensity)
        self.assertEqual(pl_engine.natoms, 2)
        self.assertEqual(len(pl_engine.omega_range), 3)

        # 3. Test MSONable Serialization Dict Round-Trip
        serialized_dict = pl_engine.as_dict()
        self.assertEqual(serialized_dict["@class"], "Photoluminescence")

        reconstructed = Photoluminescence.from_dict(serialized_dict)
        self.assertAlmostEqual(reconstructed.EZPL, pl_engine.EZPL)
        np.testing.assert_array_almost_equal(reconstructed.frequencies, pl_engine.frequencies)
        np.testing.assert_array_almost_equal(reconstructed.masses, pl_engine.masses)

        # 4. Verify Plot Dispatch Interface Routine
        mock_plotter_instance = MagicMock()
        mock_plotter_class.return_value = mock_plotter_instance

        pl_engine.generate_plots(out_dir=OUTPUT_TEST_DIR, fig_format="png")
        self.assertTrue(mock_plotter_instance.plot_penergy_vs_pmode.called)
        self.assertTrue(mock_plotter_instance.plot_intensity_vs_penergy.called)


class TestVibrationalSpectra1D(unittest.TestCase):
    """TestSuite covering the 1D Configuration Coordinate lineshape engine."""

    def setUp(self):
        """Set up standard 1D parabola tracking variables."""
        self.spectra_1d = VibrationalSpectra1D(
            EZPL=1.95,
            w1_meV=55.0,   # Excited state frequency
            w2_meV=50.0,   # Ground state frequency
            DQ=1.2,        # amu^(1/2) * Å
            T=300.0,       # Kelvin
            E0=1.0,        # Min Energy grid
            dE=0.01,       # Step
            M=200          # Array Elements Count
        )

    def test_initialization_and_lineshape_pipeline(self):
        """Verifies mathematical initialization limits and numerical lineshape generation loops."""
        # Check derived property conversion scales
        self.assertAlmostEqual(self.spectra_1d.w1, 0.055)
        self.assertAlmostEqual(self.spectra_1d.w2, 0.050)
        self.assertGreater(self.spectra_1d.Erel1, 0.0)
        self.assertGreater(self.spectra_1d.Erel2, 0.0)

        # Compute full structural lineshape spectrum
        self.spectra_1d.compute_lineshape()

        # Check output population maps
        self.assertIn("energies", self.spectra_1d.spectral_data)
        self.assertIn("dosw3", self.spectra_1d.spectral_data)
        self.assertEqual(len(self.spectra_1d.spectral_data["dosw3"]), 200)

        # 1D Spectrum should be properly normalized via dE integration steps
        dosw3_array = np.array(self.spectra_1d.spectral_data["dosw3"])
        integrated_sum = np.sum(dosw3_array) * self.spectra_1d.dE
        self.assertAlmostEqual(integrated_sum, 1.0, places=4)

    def test_metrics_extraction(self):
        """Ensures peak tracking limits and FWHM calculations evaluate safely."""
        self.spectra_1d.compute_lineshape()

        # Peak position queries
        peak_e, peak_i = self.spectra_1d.get_peak_position()
        self.assertLess(peak_e, self.spectra_1d.EZPL)  # Emission transitions shift down
        self.assertGreater(peak_i, 0.0)

        # FWHM boundaries processing checks
        fwhm = self.spectra_1d.get_fwhm()
        self.assertGreater(fwhm, 0.0)

    def test_exception_guards_on_uncomputed_states(self):
        """Guards against requesting metrics outputs before executing solver pipelines."""
        fresh_spectra = VibrationalSpectra1D(1.95, 50, 50, 1.0, 300, 1.0, 0.01, 100)
        with self.assertRaises(ValueError):
            fresh_spectra.get_peak_position()
        with self.assertRaises(ValueError):
            fresh_spectra.get_fwhm()
        with self.assertRaises(ValueError):
            fresh_spectra.plot_lineshape()

    def test_results_serialization_and_plotting_io(self):
        """Validates that text output serialization handles file generation correctly without crashing."""
        self.spectra_1d.compute_lineshape()
        
        overlap_json = OUTPUT_TEST_DIR / "test_overlap.json"
        lineshape_json = OUTPUT_TEST_DIR / "test_lineshape.json"
        plot_img = OUTPUT_TEST_DIR / "test_1d_lineshape.png"

        try:
            self.spectra_1d.save_results(str(overlap_json), str(lineshape_json))
            self.assertTrue(overlap_json.exists())
            self.assertTrue(lineshape_json.exists())

            self.spectra_1d.plot_lineshape(save_file=str(plot_img))
            self.assertTrue(plot_img.exists())
        finally:
            # File system cleanup
            for path in [overlap_json, lineshape_json, plot_img]:
                if path.exists():
                    path.unlink()


class TestConfigurationCoordinateDiagram(unittest.TestCase):
    """TestSuite covering potential energy surface structural generation loops and parabolic wells fitting."""

    def setUp(self):
        """Generate dummy Pymatgen structures mapping out a synthetic Diamond 1x1x1 crystal model."""
        lattice = Lattice.cubic(3.567)
        # Ground structure
        self.struct_g = Structure(
            lattice, 
            ["C", "C"], 
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        )
        # Distorted excited structure
        self.struct_e = Structure(
            lattice, 
            ["C", "C"], 
            [[0.01, 0.0, 0.0], [0.24, 0.25, 0.25]]
        )

        MOCK_INPUT_DIR_G.mkdir(parents=True, exist_ok=True)
        MOCK_INPUT_DIR_E.mkdir(parents=True, exist_ok=True)
        for f in ["INCAR", "POTCAR", "KPOINTS"]:
            (MOCK_INPUT_DIR_G / f).write_text(f"Ground {f} Dummy Text Content", encoding="utf-8")
            (MOCK_INPUT_DIR_E / f).write_text(f"Excited {f} Dummy Text Content", encoding="utf-8")

    def tearDown(self):
        """Clean up filesystem directories generated during test runs."""
        for directory in [MOCK_INPUT_DIR_G, MOCK_INPUT_DIR_E, OUTPUT_TEST_DIR]:
            if directory.exists():
                for item in directory.rglob("*"):
                    if item.is_file():
                        item.unlink()
                if directory.is_dir():
                    try:
                        os.removedirs(directory)
                    except OSError:
                        pass

    def test_structural_interpolation(self):
        """Validates configuration coordinates interpolation bounds tracking."""
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        self.assertGreater(ccd.dQ, 0.0)

        # Interpolate a sequence array spanning 3 displacement coordinates
        displacements = [-0.2, 0.5, 1.2]
        g_structs, e_structs = ccd.generate_structures(displacements, remove_zero=True)

        self.assertEqual(len(g_structs), 3)
        self.assertEqual(len(e_structs), 3)
        self.assertIsInstance(g_structs[0], Structure)

    @patch("defectpl.defectpl.copyfile")
    def test_calculation_trees_setup(self, mock_copyfile):
        """Verifies configuration paths logic writes standard input directory structures cleanly."""
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        displacements = [-0.5, 0.5, 1.5]
        
        ccd.setup_calculations(
            displacements=displacements,
            output_dir=OUTPUT_TEST_DIR,
            ground_input_dir=MOCK_INPUT_DIR_G,
            excited_input_dir=MOCK_INPUT_DIR_E
        )

        # Verify target structure layouts are exported seamlessly
        self.assertTrue((OUTPUT_TEST_DIR / "ground" / "0" / "POSCAR").exists())
        self.assertTrue((OUTPUT_TEST_DIR / "excited" / "0" / "POSCAR").exists())
        self.assertTrue(mock_copyfile.called)

    @patch("pymatgen.io.vasp.outputs.Vasprun")
    @patch("defectpl.utils.get_q_from_structure")
    def test_profile_extraction_and_cc_analysis(self, mock_get_q, mock_vasprun_class):
        """Tests integrated end-to-end PES curve profiling and vertical transitions calculation."""
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        
        # Build mock Vasprun data maps
        mock_vr = MagicMock()
        mock_vr.structures = [self.struct_g]
        mock_vr.final_energy = -15.5
        mock_vasprun_class.return_value = mock_vr

        # Define synthetic coordinate output rules
        mock_get_q.side_effect = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]

        dummy_paths = ["run1.xml", "run2.xml", "run3.xml"]
        
        # Test extraction metrics pipeline
        q_vals, energies = ccd.extract_pes_profile(dummy_paths)
        self.assertEqual(len(q_vals), 3)
        np.testing.assert_array_almost_equal(energies, np.zeros(3))

        # Test full diagram curves tracking loops with hardcoded mock curves outputs
        with patch("defectpl.utils.get_omega_from_pes") as mock_get_omega:
            mock_get_omega.return_value = 0.045  # eV
            
            w_g, w_e = ccd.analyze_ccd(
                ground_vaspruns=dummy_paths,
                excited_vaspruns=dummy_paths,
                dE=2.0,
                plot=False
            )
            self.assertEqual(w_g, 0.045)
            self.assertEqual(w_e, 0.045)

            # 4. Check Franck-Condon and Vertical transitions scaling mathematics
            # Overwrite dQ manually to guarantee controlled algebraic test conditions
            ccd.dQ = 1.0  
            e_abs, e_em, fc_e, fc_g = ccd.estimate_vertical_transitions(
                ground_omega=0.05, 
                excited_omega=0.05, 
                dE=2.0
            )
            
            self.assertGreater(e_abs, 2.0)  # Absorption is higher than ZPL gap energy
            self.assertLess(e_em, 2.0)     # Emission is lower than ZPL gap energy
            self.assertAlmostEqual(e_abs - fc_e, 2.0)


if __name__ == "__main__":
    unittest.main()