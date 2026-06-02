# -*- coding: utf-8 -*-
"""
Unit tests tracking execution routes across the defectpl CLI application layer, 
focusing on Displacement Mode, Force Mode, and custom standalone Plotting subcommands.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from click.testing import CliRunner

# Import our root entry point group from the cli file
from defectpl.cli import main


@pytest.fixture
def cli_runner():
    """Provides a fresh Click CLI runner instance for clean isolated execution tracks."""
    return CliRunner()


@pytest.fixture
def mock_dependencies():
    """
    Mocks backend calculations, file parsers, and plotting pipelines to isolate 
    the application interface layer from heavy file system IO or structural mathematics.
    """
    with patch("defectpl.phonon.read_band_yaml") as mock_read, \
         patch("pymatgen.core.Structure.from_file") as mock_struct, \
         patch("defectpl.vasp_wrapper.calc_dR") as mock_dr, \
         patch("defectpl.vasp_wrapper.prepare_dF_files") as mock_df, \
         patch("defectpl.defectpl.Photoluminescence") as mock_pl, \
         patch("defectpl.plot.Plotter") as mock_plotter, \
         patch("monty.serialization.dumpfn") as mock_dump, \
         patch("monty.serialization.loadfn") as mock_load:
        
        # Setup typical dummy outputs from mocked structures
        mock_read.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_dr.return_value = MagicMock()
        mock_df.return_value = MagicMock()
        
        yield {
            "read_band_yaml": mock_read,
            "structure_from_file": mock_struct,
            "calc_dR": mock_dr,
            "prepare_dF_files": mock_df,
            "Photoluminescence": mock_pl,
            "Plotter": mock_plotter,
            "dumpfn": mock_dump,
            "loadfn": mock_load,
        }


def test_pl_displacement_mode_basic(cli_runner, mock_dependencies):
    """Verifies that pl displacement command resolves cleanly with basic parameters."""
    with cli_runner.isolated_filesystem():
        # Create dummy physical files so click validation checks clear out
        Path("band.yaml").touch()
        Path("CONTCAR_gs").touch()
        Path("CONTCAR_es").touch()

        result = cli_runner.invoke(
            main,
            [
                "pl", "displacement",
                "--band_yaml", "band.yaml",
                "--contcar_gs", "CONTCAR_gs",
                "--contcar_es", "CONTCAR_es",
                "--ezpl", "2.1",
                "--gamma", "1.5",
            ]
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert "Initializing multi-mode PL calculation via Displacement Mode..." in result.output
        
        # Verify core parameters mapped down into structural physics engines
        mock_dependencies["Photoluminescence"].assert_called_once()
        kwargs = mock_dependencies["Photoluminescence"].call_args[1]
        assert kwargs["EZPL"] == 2.1
        assert kwargs["gamma"] == 1.5


def test_pl_displacement_mode_with_json_and_plots(cli_runner, mock_dependencies):
    """Verifies that serialization dumps and plot customization triggers compile down correctly."""
    with cli_runner.isolated_filesystem():
        Path("band.yaml").touch()
        Path("CONTCAR_gs").touch()
        Path("CONTCAR_es").touch()

        # Instantiate mock engine wrapper instance
        pl_instance = MagicMock()
        mock_dependencies["Photoluminescence"].return_value = pl_instance

        result = cli_runner.invoke(
            main,
            [
                "pl", "displacement",
                "--json_out", "state_dump.json",
                "--plot_all",
                "--iylim", "0,1.5",
                "--max_freq", "85.0",
            ]
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Check if monty serialization protocol was initiated on our calculations engine
        mock_dependencies["dumpfn"].assert_called_once_with(pl_instance, "state_dump.json", indent=2)
        
        # Check custom axis layout scaling mappings
        pl_instance.generate_plots.assert_called_once_with(
            out_dir="./",
            fig_format="pdf",
            iylim=[0.0, 1.5],
            max_freq=85.0
        )


def test_pl_force_mode_basic(cli_runner, mock_dependencies):
    """Verifies that pl force command resolves clean and pulls out file logs cleanly."""
    with cli_runner.isolated_filesystem():
        Path("band.yaml").touch()
        Path("OUTCAR_gs").touch()
        Path("OUTCAR_es").touch()

        result = cli_runner.invoke(
            main,
            [
                "pl", "force",
                "--band_yaml", "band.yaml",
                "--outcar_gs", "OUTCAR_gs",
                "--outcar_es", "OUTCAR_es",
                "--ezpl", "1.85",
            ]
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert "Initializing multi-mode PL calculation via Force Mode..." in result.output
        mock_dependencies["prepare_dF_files"].assert_called_once_with("OUTCAR_gs", "OUTCAR_es")


def test_standalone_plot_command(cli_runner, mock_dependencies):
    """Verifies the standalone plot command safely loads JSON records and delegates to the Plotter."""
    with cli_runner.isolated_filesystem():
        Path("saved_state.json").touch()

        pl_mock_obj = MagicMock()
        mock_dependencies["loadfn"].return_value = pl_mock_obj

        plotter_instance = MagicMock()
        mock_dependencies["Plotter"].return_value = plotter_instance

        # Patch isinstance inside defectpl.cli to bypass mock-type limitations
        with patch("defectpl.cli.isinstance", return_value=True):
            result = cli_runner.invoke(
                main,
                [
                    "plot", "saved_state.json",
                    "--type", "all",
                    "--iylim", "0.1,1.0",
                    "--max_freq", "60.0"
                ]
            )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        mock_dependencies["loadfn"].assert_called_once_with("saved_state.json")
        mock_dependencies["Plotter"].assert_called_once_with(pl_mock_obj)
        
        # Confirm individual plot variations have run with custom overrides
        plotter_instance.plot_intensity_vs_penergy.assert_called_once_with(out_dir="./", fig_format="pdf", iylim=[0.1, 1.0])
        plotter_instance.plot_penergy_vs_pmode.assert_called_once_with(out_dir="./", fig_format="pdf", max_freq=60.0)


def test_standalone_plot_type_filtering(cli_runner, mock_dependencies):
    """Ensures standalone plotting command skips irrelevant subplots when a specific type is requested."""
    with cli_runner.isolated_filesystem():
        Path("saved_state.json").touch()
        
        pl_mock_obj = MagicMock()
        mock_dependencies["loadfn"].return_value = pl_mock_obj
        
        plotter_instance = MagicMock()
        mock_dependencies["Plotter"].return_value = plotter_instance

        # Patch isinstance inside defectpl.cli to bypass mock-type limitations
        with patch("defectpl.cli.isinstance", return_value=True):
            result = cli_runner.invoke(
                main,
                [
                    "plot", "saved_state.json",
                    "--type", "intensity"
                ]
            )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Intensity should be called, but Mode plots should be skipped completely
        plotter_instance.plot_intensity_vs_penergy.assert_called_once()
        plotter_instance.plot_penergy_vs_pmode.assert_not_called()