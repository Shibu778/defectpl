# -*- coding: utf-8 -*-
"""
Unit test suite for the Defect Optical Properties Engine (DefectPL) core module.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.core import Structure, Lattice

# Import target classes from the code module layout cleanly
from defectpl.defectpl import (
    Photoluminescence,
    VibrationalSpectra1D,
    ConfigurationCoordinateDiagram,
)


class TestPhotoluminescence(unittest.TestCase):

    def setUp(self):
        """Set up dummy array data configurations for Photoluminescence testing."""
        self.frequencies = np.array([0.02, 0.04, 0.06])  # eV
        self.eigenvectors = np.ones((3, 2, 3))  # (nmodes, natoms, 3)
        self.masses = np.array([12.011, 14.007])  # C and N atom mock masses
        self.dR = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        self.dF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.EZPL = 2.5
        self.gamma = 0.01

    @patch("defectpl.defectpl.utils", autospec=True)
    def test_initialization_and_computation_pipeline(self, mock_utils):
        """Verify the post-init lifecycle triggers calculations and assignments correctly."""
        mock_utils.calc_delR.return_value = 0.1414
        mock_utils.calc_delQ.return_value = 0.55
        mock_utils.calc_qks.return_value = np.array([0.1, 0.2, 0.3])
        mock_utils.calc_Sks.return_value = np.array([1.0, 0.5, 0.2])
        mock_utils.calc_IPR.return_value = np.array([1.5, 1.5, 1.5])
        mock_utils.calc_S_omega.return_value = np.linspace(0, 1, 100)
        mock_utils.calc_St.return_value = np.linspace(0, 1, 100)
        mock_utils.calc_Gts.return_value = np.linspace(0, 1, 100)
        mock_utils.calc_Spectrum_Intensity.return_value = (
            np.ones(100),
            np.ones(100) * 5,
        )

        pl = Photoluminescence(
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            masses=self.masses,
            dR=self.dR,
            dF=None,
            EZPL=self.EZPL,
            gamma=self.gamma,
            resolution=10,
            max_energy=2.0,
        )

        self.assertEqual(pl.natoms, 2)
        self.assertAlmostEqual(pl.HR_factor, 1.7)
        self.assertAlmostEqual(pl.DW_factor, np.exp(-1.7))
        mock_utils.calc_qks.assert_called_once()
        mock_utils.calc_qks_force_mode.assert_not_called()

    @patch("defectpl.defectpl.utils", autospec=True)
    def test_force_mode_fallback_switch(self, mock_utils):
        """Ensure dF triggers force-mode calculation blocks over structural changes."""
        mock_utils.calc_qks_force_mode.return_value = np.array([0.1, 0.2, 0.3])
        mock_utils.calc_Sks.return_value = np.array([1.0, 0.5, 0.2])
        mock_utils.calc_Spectrum_Intensity.return_value = (np.ones(10), np.ones(10))

        active_dF = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        pl = Photoluminescence(
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            masses=self.masses,
            dR=None,
            dF=active_dF,
            EZPL=self.EZPL,
            gamma=self.gamma,
        )
        mock_utils.calc_qks_force_mode.assert_called_once()

    @patch("defectpl.defectpl.utils", autospec=True)
    def test_asymmetric_serialization_and_deserialization(self, mock_utils):
        """Test both standard loading (instant state restore) and expensive regeneration methods."""
        mock_utils.calc_Spectrum_Intensity.return_value = (np.ones(10), np.ones(10))
        mock_utils.calc_Sks.return_value = np.array([1.0, 0.5, 0.2])

        pl = Photoluminescence(
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            masses=self.masses,
            dR=self.dR,
            dF=None,
            EZPL=self.EZPL,
            gamma=self.gamma,
        )

        d = pl.as_dict()
        self.assertIn("intensity", d)
        self.assertIn("HR_factor", d)

        mock_utils.calc_qks.reset_mock()

        # 1. Test standard from_dict: loads pre-computed qks/Sks from dict, skips calc_qks
        restored_fast = Photoluminescence.from_dict(d)
        self.assertIsInstance(restored_fast, Photoluminescence)
        mock_utils.calc_qks.assert_not_called()

        # 2. Test from_dict_expensive: recomputes everything from scratch via __post_init__
        mock_utils.calc_qks.reset_mock()
        restored_expensive = Photoluminescence.from_dict_expensive(d)
        self.assertIsInstance(restored_expensive, Photoluminescence)
        mock_utils.calc_qks.assert_called_once()

    @patch("defectpl.defectpl.Plotter")
    @patch("defectpl.defectpl.utils", autospec=True)
    def test_generate_plots_routing(self, mock_utils, mock_plotter_cls):
        """Assert plot rendering directives are safely forwarded to the plotting engine."""
        mock_plotter = mock_plotter_cls.return_value
        mock_utils.calc_Spectrum_Intensity.return_value = (np.ones(10), np.ones(10))
        mock_utils.calc_Sks.return_value = np.array([1.0, 0.5, 0.2])

        pl = Photoluminescence(
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            masses=self.masses,
            dR=self.dR,
            dF=None,
            EZPL=self.EZPL,
            gamma=self.gamma,
        )

        pl.generate_plots(out_dir="/tmp/test_plots")
        self.assertTrue(mock_plotter.plot_penergy_vs_pmode.called)
        self.assertTrue(mock_plotter.plot_intensity_vs_penergy.called)


class TestVibrationalSpectra1D(unittest.TestCase):

    def setUp(self):
        """Initialize configurations for 1D Harmonic Oscillator limits."""
        self.spectra = VibrationalSpectra1D(
            EZPL=2.0, w1_meV=40.0, w2_meV=35.0, DQ=0.5, T=10.0, E0=1.5, dE=0.01, M=100
        )

    @patch("defectpl.defectpl.utils.calculate_overlap_element")
    def test_overlap_matrix_population(self, mock_calc):
        """Verify matrix instantiation maps parameters down cleanly."""
        mock_calc.return_value = 0.05
        self.spectra.compute_overlap_matrix()

        expected_shape = (self.spectra.NN1 + 1, self.spectra.NN2 + 1)
        self.assertEqual(self.spectra.overlap_matrix.shape, expected_shape)
        self.assertAlmostEqual(self.spectra.overlap_matrix[0, 0], 0.05)

    def test_compute_lineshape_execution_and_metrics(self):
        """Verify vector calculations populate spectra metrics without failing."""
        self.spectra.overlap_data = {
            "contributions": [0.5, 0.5],
            "energies": [2.0, 1.95],
        }
        self.spectra.compute_lineshape()

        self.assertIn("energies", self.spectra.spectral_data)
        self.assertIn("dosw3", self.spectra.spectral_data)
        self.assertEqual(len(self.spectra.spectral_data["energies"]), 100)

        peak_e, _ = self.spectra.get_peak_position()
        fwhm = self.spectra.get_fwhm()
        self.assertIsInstance(peak_e, float)
        self.assertIsInstance(fwhm, float)


class TestConfigurationCoordinateDiagram(unittest.TestCase):

    def setUp(self):
        """Construct mock crystal lattices to avoid disk-bound structural parsing hooks."""
        lattice = Lattice.cubic(3.5)
        self.struct_g = Structure(lattice, ["C"], [[0, 0, 0]])
        self.struct_e = Structure(lattice, ["C"], [[0.05, 0.0, 0.0]])

    @patch("defectpl.defectpl.calc_delta_Q")
    def test_ccd_initialization(self, mock_dq):
        """Assert coordinates difference profiles resolve during initiation tasks."""
        mock_dq.return_value = 0.25
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        self.assertEqual(ccd.dQ, 0.25)

    @patch("defectpl.defectpl.calc_delta_Q")
    def test_structure_interpolation_generation(self, mock_dq):
        """Validate structural generation loops generate interpolation steps cleanly."""
        mock_dq.return_value = 0.2
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        g_list, e_list = ccd.generate_structures(displacements=[-0.5, 0.5])

        self.assertEqual(len(g_list), 2)
        self.assertEqual(len(e_list), 2)

    @patch("pymatgen.io.vasp.outputs.Vasprun")
    @patch("defectpl.defectpl.get_q_from_structure")
    @patch("defectpl.defectpl.calc_delta_Q")
    def test_extract_pes_profile(self, mock_dq, mock_get_q, mock_vasprun_cls):
        """Assert file outputs parse configurations across coordinated targets safely."""
        mock_dq.return_value = 0.1
        mock_get_q.return_value = 1.2

        mock_vr = MagicMock()
        mock_vr.structures = [self.struct_g]
        mock_vr.final_energy = -10.5
        mock_vasprun_cls.return_value = mock_vr

        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)
        q_vals, e_vals = ccd.extract_pes_profile(["path/to/vasprun.xml"])

        np.testing.assert_array_equal(q_vals, [1.2])
        np.testing.assert_array_equal(e_vals, [0.0])

    @patch("defectpl.defectpl.calc_delta_Q")
    def test_estimate_vertical_transitions(self, mock_dq):
        """Ensure energy transition thresholds are accurately computed using physical constants."""
        mock_dq.return_value = 0.3
        ccd = ConfigurationCoordinateDiagram(self.struct_g, self.struct_e)

        with self.assertRaises(ValueError):
            ccd.estimate_vertical_transitions(
                ground_omega=0.0, excited_omega=0.05, dE=2.0
            )

        e_abs, e_em, fc_e, fc_g = ccd.estimate_vertical_transitions(
            ground_omega=0.04, excited_omega=0.04, dE=2.5
        )
        self.assertTrue(e_abs > 2.5)
        self.assertTrue(e_em < 2.5)


if __name__ == "__main__":
    unittest.main()
