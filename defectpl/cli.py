# -*- coding: utf-8 -*-
"""
Command Line Interface (CLI) application layer managing execution workflows
for the defectpl suite, covering photoluminescence lineshapes, structural
displacements, Configuration Coordinate Diagrams (CCD), and phonon properties.
"""

import json
import traceback
from pathlib import Path
import click

_verbose = False


def _set_verbose(ctx, param, value):
    global _verbose
    _verbose = value


@click.group()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    is_eager=True,
    expose_value=False,
    callback=_set_verbose,
    help="Show the full error traceback instead of a condensed message.",
)
def main():
    """defectpl command utility suite for defect photoluminescence modeling."""
    pass


# =====================================================================
# PHOTOLUMINESCENCE ENGINE COMMANDS (DISPLACEMENT & FORCE MODES)
# =====================================================================


@click.group(name="pl")
def pl_group():
    """Execute high-level defect photoluminescence multi-mode spectral profile engines."""
    pass


def _parse_sigma(sigma_str: str):
    """Parse '--sigma 6e-3' (scalar) or '--sigma 3e-3,8e-3' (tuple) into float or tuple."""
    parts = [s.strip() for s in sigma_str.split(",")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise click.BadParameter(
        "sigma must be a single float (e.g. '6e-3') or two comma-separated floats "
        "(e.g. '3e-3,8e-3') for frequency-dependent broadening."
    )


@pl_group.command(name="displacement")
@click.option(
    "--band_yaml",
    default="./band.yaml",
    type=click.Path(exists=True),
    help="Phonopy band.yaml configuration file tracking destination path.",
)
@click.option(
    "--contcar_gs",
    default="./CONTCAR_gs",
    type=click.Path(exists=True),
    help="Pymatgen readable Ground State equilibrium configuration structure file path.",
)
@click.option(
    "--contcar_es",
    default="./CONTCAR_es",
    type=click.Path(exists=True),
    help="Pymatgen readable Excited State equilibrium configuration structure file path.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output directory path endpoint to drop calculated results database records.",
)
@click.option(
    "--ezpl",
    default=1.95,
    type=float,
    help="Energy value designating Zero-Phonon Line baseline transitions boundary in eV.",
)
@click.option(
    "--gamma",
    default=2.0,
    type=float,
    help="Lorentzian ZPL broadening in meV.",
)
@click.option(
    "--temperature",
    default=0.0,
    type=float,
    help="Lattice temperature in K for Bose-Einstein thermal weighting (0 = T=0 limit).",
)
@click.option(
    "--sigma",
    "sigma_str",
    default="6e-3",
    help=(
        "Gaussian broadening in eV: scalar '6e-3' for uniform, or two comma-separated "
        "values '3e-3,8e-3' for frequency-dependent (low-freq,high-freq) broadening."
    ),
)
@click.option(
    "--resolution",
    default=1000,
    type=int,
    help="Number of spectral grid points per eV.",
)
@click.option(
    "--max_energy",
    default=5.0,
    type=float,
    help="Upper energy axis limit in eV.",
)
@click.option(
    "--json_out",
    default=None,
    help="Optional output path to serialize the complete Photoluminescence data model as a JSON file.",
)
@click.option(
    "--plot_all",
    is_flag=True,
    default=False,
    help="If flagged, automatically spawns all downstream visualization graphics profiles.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension target layout standard (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated limits for the intensity plot y-axis (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency limit for mode analysis plots (in meV).",
)
def pl_displacement(
    band_yaml,
    contcar_gs,
    contcar_es,
    out_dir,
    ezpl,
    gamma,
    temperature,
    sigma_str,
    resolution,
    max_energy,
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run PL calculations using atomic structural shifts (Displacement Mode)."""
    from defectpl.phonon import read_band_yaml
    from pymatgen.core import Structure
    from defectpl.io.vasp import calc_dR
    from defectpl.defectpl import Photoluminescence
    from monty.serialization import dumpfn

    try:
        sigma = _parse_sigma(sigma_str)
        click.echo("Initializing multi-mode PL calculation via Displacement Mode...")
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        struct_gs = Structure.from_file(contcar_gs)
        struct_es = Structure.from_file(contcar_es)
        dR = calc_dR(struct_gs, struct_es)

        pl_engine = Photoluminescence(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=dR,
            dF=None,
            EZPL=ezpl,
            gamma=gamma,
            resolution=resolution,
            max_energy=max_energy,
            sigma=sigma,
            temperature=temperature,
        )
        click.echo("Photoluminescence engine data properties calculated successfully.")
        click.echo(f"  HR factor      : {pl_engine.HR_factor:.4f}")
        click.echo(f"  DW factor      : {pl_engine.DW_factor:.4f}")
        click.echo(f"  Temperature    : {temperature} K")
        click.echo(f"  C_total        : {pl_engine.C_total:.4f}")

        if json_out:
            dumpfn(pl_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoluminescence class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            pl_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )
            click.echo(f"All plots written to {out_dir}")

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


@pl_group.command(name="force")
@click.option(
    "--band_yaml",
    default="./band.yaml",
    type=click.Path(exists=True),
    help="Phonopy band.yaml configuration file tracking destination path.",
)
@click.option(
    "--outcar_gs",
    default="./OUTCAR_gs",
    type=click.Path(exists=True),
    help="VASP OUTCAR calculation log file tracking ground state atomic forces.",
)
@click.option(
    "--outcar_es",
    default="./OUTCAR_es",
    type=click.Path(exists=True),
    help="VASP OUTCAR calculation log file tracking excited state vertical forces.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output directory path endpoint to drop calculated results database records.",
)
@click.option(
    "--ezpl",
    default=1.95,
    type=float,
    help="Energy value designating Zero-Phonon Line baseline transitions boundary in eV.",
)
@click.option(
    "--gamma",
    default=2.0,
    type=float,
    help="Lorentzian ZPL broadening in meV.",
)
@click.option(
    "--temperature",
    default=0.0,
    type=float,
    help="Lattice temperature in K for Bose-Einstein thermal weighting (0 = T=0 limit).",
)
@click.option(
    "--sigma",
    "sigma_str",
    default="6e-3",
    help=(
        "Gaussian broadening in eV: scalar '6e-3' for uniform, or two comma-separated "
        "values '3e-3,8e-3' for frequency-dependent (low-freq,high-freq) broadening."
    ),
)
@click.option(
    "--resolution",
    default=1000,
    type=int,
    help="Number of spectral grid points per eV.",
)
@click.option(
    "--max_energy",
    default=5.0,
    type=float,
    help="Upper energy axis limit in eV.",
)
@click.option(
    "--json_out",
    default=None,
    help="Optional output path to serialize the complete Photoluminescence data model as a JSON file.",
)
@click.option(
    "--plot_all",
    is_flag=True,
    default=False,
    help="If flagged, automatically spawns all downstream visualization graphics profiles.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension target layout standard (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated limits for the intensity plot y-axis (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency limit for mode analysis plots (in meV).",
)
def pl_force(
    band_yaml,
    outcar_gs,
    outcar_es,
    out_dir,
    ezpl,
    gamma,
    temperature,
    sigma_str,
    resolution,
    max_energy,
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run PL calculations using force-difference vectors at vertical excitation (Force Mode)."""
    from defectpl.phonon import read_band_yaml
    from defectpl.io.vasp import prepare_dF_files
    from defectpl.defectpl import Photoluminescence
    from monty.serialization import dumpfn

    try:
        sigma = _parse_sigma(sigma_str)
        click.echo("Initializing multi-mode PL calculation via Force Mode...")
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        dF = prepare_dF_files(outcar_gs, outcar_es)

        pl_engine = Photoluminescence(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=None,
            dF=dF,
            EZPL=ezpl,
            gamma=gamma,
            resolution=resolution,
            max_energy=max_energy,
            sigma=sigma,
            temperature=temperature,
        )
        click.echo("Photoluminescence engine data properties calculated successfully.")
        click.echo(f"  HR factor      : {pl_engine.HR_factor:.4f}")
        click.echo(f"  DW factor      : {pl_engine.DW_factor:.4f}")
        click.echo(f"  Temperature    : {temperature} K")
        click.echo(f"  C_total        : {pl_engine.C_total:.4f}")

        if json_out:
            dumpfn(pl_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoluminescence class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            pl_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )
            click.echo(f"All plots written to {out_dir}")

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


@pl_group.command(name="from-json")
@click.argument("json_file", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    default="./",
    help="Directory to write plots.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated y-axis limits for intensity plot (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency for mode plots (meV).",
)
def pl_from_json(json_file, out_dir, fig_format, iylim, max_freq):
    """Restore a saved Photoluminescence JSON and regenerate all plots.

    \b
        defectpl pl from-json pl_results.json --out_dir plots/ --fig_format png
    """
    from defectpl.defectpl import Photoluminescence
    from monty.serialization import loadfn

    try:
        click.echo(f"Loading: {json_file}")
        pl_engine = loadfn(json_file)
        if not isinstance(pl_engine, Photoluminescence):
            raise ValueError("JSON does not contain a Photoluminescence object.")

        click.echo(
            f"  HR factor : {pl_engine.HR_factor:.4f}  |  "
            f"EZPL : {pl_engine.EZPL} eV  |  T : {pl_engine.temperature} K"
        )
        parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        pl_engine.generate_plots(
            out_dir=out_dir,
            fig_format=fig_format,
            iylim=parsed_iylim,
            max_freq=max_freq,
        )
        click.echo(f"All plots written to {out_dir}")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))


# =====================================================================
# PHOTOABSORPTION ENGINE COMMANDS (DISPLACEMENT & FORCE MODES)
# =====================================================================


@click.group(name="absorption")
def absorption_group():
    """Execute photoabsorption spectral calculations using excited-state phonons."""
    pass


@absorption_group.command(name="displacement")
@click.option(
    "--band_yaml",
    default="./band.yaml",
    type=click.Path(exists=True),
    help="Excited-state phonopy band.yaml (phonopy run on the ES geometry).",
)
@click.option(
    "--contcar_gs",
    default="./CONTCAR_gs",
    type=click.Path(exists=True),
    help="Pymatgen readable Ground State equilibrium configuration structure file path.",
)
@click.option(
    "--contcar_es",
    default="./CONTCAR_es",
    type=click.Path(exists=True),
    help="Pymatgen readable Excited State equilibrium configuration structure file path.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output directory path endpoint to drop calculated results database records.",
)
@click.option(
    "--ezpl",
    default=1.95,
    type=float,
    help="Energy value designating Zero-Phonon Line baseline transitions boundary in eV.",
)
@click.option(
    "--gamma",
    default=2.0,
    type=float,
    help="Lorentzian ZPL broadening in meV.",
)
@click.option(
    "--temperature",
    default=0.0,
    type=float,
    help="Lattice temperature in K for Bose-Einstein thermal weighting (0 = T=0 limit).",
)
@click.option(
    "--sigma",
    "sigma_str",
    default="6e-3",
    help=(
        "Gaussian broadening in eV: scalar '6e-3' for uniform, or two comma-separated "
        "values '3e-3,8e-3' for frequency-dependent (low-freq,high-freq) broadening."
    ),
)
@click.option(
    "--resolution",
    default=1000,
    type=int,
    help="Number of spectral grid points per eV.",
)
@click.option(
    "--max_energy",
    default=5.0,
    type=float,
    help="Upper energy axis limit in eV.",
)
@click.option(
    "--json_out",
    default=None,
    help="Optional output path to serialize the complete Photoabsorption data model as a JSON file.",
)
@click.option(
    "--plot_all",
    is_flag=True,
    default=False,
    help="If flagged, automatically spawns all downstream visualization graphics profiles.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension target layout standard (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated limits for the absorption plot y-axis (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency limit for mode analysis plots (in meV).",
)
def absorption_displacement(
    band_yaml,
    contcar_gs,
    contcar_es,
    out_dir,
    ezpl,
    gamma,
    temperature,
    sigma_str,
    resolution,
    max_energy,
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run photoabsorption calculations using atomic structural shifts (Displacement Mode)."""
    from defectpl.phonon import read_band_yaml
    from pymatgen.core import Structure
    from defectpl.io.vasp import calc_dR
    from defectpl.defectpl import Photoabsorption
    from monty.serialization import dumpfn

    try:
        sigma = _parse_sigma(sigma_str)
        click.echo("Initializing photoabsorption calculation via Displacement Mode...")
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        struct_gs = Structure.from_file(contcar_gs)
        struct_es = Structure.from_file(contcar_es)
        dR = calc_dR(struct_gs, struct_es)

        abs_engine = Photoabsorption(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=dR,
            dF=None,
            EZPL=ezpl,
            gamma=gamma,
            resolution=resolution,
            max_energy=max_energy,
            sigma=sigma,
            temperature=temperature,
        )
        click.echo("Photoabsorption engine data properties calculated successfully.")
        click.echo(f"  HR factor      : {abs_engine.HR_factor:.4f}")
        click.echo(f"  DW factor      : {abs_engine.DW_factor:.4f}")
        click.echo(f"  Temperature    : {temperature} K")
        click.echo(f"  C_total        : {abs_engine.C_total:.4f}")

        if json_out:
            dumpfn(abs_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoabsorption class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            abs_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )
            click.echo(f"All plots written to {out_dir}")

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


@absorption_group.command(name="force")
@click.option(
    "--band_yaml",
    default="./band.yaml",
    type=click.Path(exists=True),
    help="Excited-state phonopy band.yaml (phonopy run on the ES geometry).",
)
@click.option(
    "--outcar_gs",
    default="./OUTCAR_gs",
    type=click.Path(exists=True),
    help="OUTCAR at ES geometry with GS charge state (for force difference).",
)
@click.option(
    "--outcar_es",
    default="./OUTCAR_es",
    type=click.Path(exists=True),
    help="OUTCAR at ES geometry with ES charge state.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output directory path endpoint to drop calculated results database records.",
)
@click.option(
    "--ezpl",
    default=1.95,
    type=float,
    help="Energy value designating Zero-Phonon Line baseline transitions boundary in eV.",
)
@click.option(
    "--gamma",
    default=2.0,
    type=float,
    help="Lorentzian ZPL broadening in meV.",
)
@click.option(
    "--temperature",
    default=0.0,
    type=float,
    help="Lattice temperature in K for Bose-Einstein thermal weighting (0 = T=0 limit).",
)
@click.option(
    "--sigma",
    "sigma_str",
    default="6e-3",
    help=(
        "Gaussian broadening in eV: scalar '6e-3' for uniform, or two comma-separated "
        "values '3e-3,8e-3' for frequency-dependent (low-freq,high-freq) broadening."
    ),
)
@click.option(
    "--resolution",
    default=1000,
    type=int,
    help="Number of spectral grid points per eV.",
)
@click.option(
    "--max_energy",
    default=5.0,
    type=float,
    help="Upper energy axis limit in eV.",
)
@click.option(
    "--json_out",
    default=None,
    help="Optional output path to serialize the complete Photoabsorption data model as a JSON file.",
)
@click.option(
    "--plot_all",
    is_flag=True,
    default=False,
    help="If flagged, automatically spawns all downstream visualization graphics profiles.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension target layout standard (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated limits for the absorption plot y-axis (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency limit for mode analysis plots (in meV).",
)
def absorption_force(
    band_yaml,
    outcar_gs,
    outcar_es,
    out_dir,
    ezpl,
    gamma,
    temperature,
    sigma_str,
    resolution,
    max_energy,
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run photoabsorption calculations using force-difference vectors (Force Mode)."""
    from defectpl.phonon import read_band_yaml
    from defectpl.io.vasp import prepare_dF_files
    from defectpl.defectpl import Photoabsorption
    from monty.serialization import dumpfn

    try:
        sigma = _parse_sigma(sigma_str)
        click.echo("Initializing photoabsorption calculation via Force Mode...")
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        dF = prepare_dF_files(outcar_gs, outcar_es)

        abs_engine = Photoabsorption(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=None,
            dF=dF,
            EZPL=ezpl,
            gamma=gamma,
            resolution=resolution,
            max_energy=max_energy,
            sigma=sigma,
            temperature=temperature,
        )
        click.echo("Photoabsorption engine data properties calculated successfully.")
        click.echo(f"  HR factor      : {abs_engine.HR_factor:.4f}")
        click.echo(f"  DW factor      : {abs_engine.DW_factor:.4f}")
        click.echo(f"  Temperature    : {temperature} K")
        click.echo(f"  C_total        : {abs_engine.C_total:.4f}")

        if json_out:
            dumpfn(abs_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoabsorption class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            abs_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )
            click.echo(f"All plots written to {out_dir}")

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


@absorption_group.command(name="from-json")
@click.argument("json_file", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    default="./",
    help="Directory to write plots.",
)
@click.option(
    "--fig_format",
    default="pdf",
    help="Export graphic file extension (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated y-axis limits for absorption plot (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Maximum phonon frequency for mode plots (meV).",
)
def absorption_from_json(json_file, out_dir, fig_format, iylim, max_freq):
    """Restore a saved Photoabsorption JSON and regenerate all plots.

    \b
        defectpl absorption from-json abs_results.json --out_dir plots/ --fig_format png
    """
    from defectpl.defectpl import Photoabsorption
    from monty.serialization import loadfn

    try:
        click.echo(f"Loading: {json_file}")
        abs_engine = loadfn(json_file)
        if not isinstance(abs_engine, Photoabsorption):
            raise ValueError("JSON does not contain a Photoabsorption object.")

        click.echo(
            f"  HR factor : {abs_engine.HR_factor:.4f}  |  "
            f"EZPL : {abs_engine.EZPL} eV  |  T : {abs_engine.temperature} K"
        )
        parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        abs_engine.generate_plots(
            out_dir=out_dir,
            fig_format=fig_format,
            iylim=parsed_iylim,
            max_freq=max_freq,
        )
        click.echo(f"All plots written to {out_dir}")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))


# =====================================================================
# INDIVIDUAL PLOT CONTROLLER COMMAND
# =====================================================================


_PLOT_TYPES = [
    "all",
    "mode",
    "ipr",
    "ipr_alkauskas",
    "loc_ratio",
    "qk",
    "hr_factor",
    "s_omega",
    "s_omega_sk",
    "s_omega_locrat",
    "s_omega_ipr",
    "s_omega_ipr_alkauskas",
    "nk",
    "c_omega",
    "intensity",
    "absorption",
]


@main.command(name="plot")
@click.argument("json_file", type=click.Path(exists=True))
@click.option(
    "--type",
    "-t",
    "plot_type",
    type=click.Choice(_PLOT_TYPES, case_sensitive=False),
    required=True,
    help=(
        "Plot to render.  Use 'all' to generate every figure.\n\n\b\n"
        "  mode              phonon energy vs mode index\n"
        "  ipr               traditional IPR vs phonon energy\n"
        "  ipr_alkauskas     Alkauskas-convention IPR vs phonon energy\n"
        "  loc_ratio         localization ratio vs phonon energy\n"
        "  qk                mode displacement q_k vs phonon energy\n"
        "  hr_factor         partial HR factor S_k vs phonon energy\n"
        "  s_omega           broadened spectral density S(omega)\n"
        "  s_omega_sk        S(omega) with S_k scatter overlay\n"
        "  s_omega_locrat    S(omega) coloured by localization ratio\n"
        "  s_omega_ipr       S(omega) coloured by traditional IPR\n"
        "  s_omega_ipr_alkauskas  S(omega) coloured by Alkauskas IPR\n"
        "  nk                Bose-Einstein phonon occupation vs phonon energy\n"
        "  c_omega           thermal spectral density C(omega,T)\n"
        "  intensity         normalised PL emission spectrum (Photoluminescence JSON only)\n"
        "  absorption        normalised absorption spectrum (Photoabsorption JSON only)\n"
        "  (for PL+absorption overlay use: defectpl overlay --pl pl.json --abs abs.json)\n"
    ),
)
@click.option(
    "--out_dir",
    default="./",
    help="Output destination directory for generated figures.",
)
@click.option(
    "--fmt",
    default="pdf",
    help="Figure file format (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated y-axis limits for the intensity plot (e.g., '0,1.2').",
)
@click.option(
    "--max_freq",
    default=None,
    type=float,
    help="Upper phonon-frequency cut-off for mode/S(omega) plots (meV).",
)
def plot_individual(json_file, plot_type, out_dir, fmt, iylim, max_freq):
    """Deserialize a saved Photoluminescence or Photoabsorption JSON and render one or all plots.

    \b
    Examples:
      defectpl plot pl.json -t intensity
      defectpl plot abs.json -t absorption --fmt png
      defectpl plot pl.json -t all --out_dir figs/
      defectpl plot pl.json -t nk
    """
    from defectpl.defectpl import Photoluminescence, Photoabsorption
    from defectpl.plot import Plotter
    from monty.serialization import loadfn

    try:
        click.echo(f"Loading: {json_file}")
        engine = loadfn(json_file)
        if not isinstance(engine, (Photoluminescence, Photoabsorption)):
            raise ValueError(
                "JSON does not contain a Photoluminescence or Photoabsorption object."
            )
        # Convenience alias used throughout the block below
        pl_engine = engine

        plotter = Plotter()
        parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
        freq_limit = (max_freq / 1000.0) if max_freq else None
        iplot_xlim = (max(0.0, pl_engine.EZPL - 2.0), pl_engine.EZPL + 1.0)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        pt = plot_type.lower()

        def _do(name, fn, *args, **kwargs):
            fn(*args, **kwargs)
            click.echo(f"  Saved: {name}")

        if pt in ("mode", "all"):
            _do(
                "mode",
                plotter.plot_penergy_vs_pmode,
                frequencies=pl_engine.frequencies,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("ipr", "all"):
            _do(
                "ipr",
                plotter.plot_ipr_vs_penergy,
                pl_engine.frequencies,
                pl_engine.iprs,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("ipr_alkauskas", "all"):
            _do(
                "ipr_alkauskas",
                plotter.plot_ipr_alkauskas_vs_penergy,
                pl_engine.frequencies,
                pl_engine.iprs_alkauskas,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("loc_ratio", "all"):
            _do(
                "loc_ratio",
                plotter.plot_loc_rat_vs_penergy,
                pl_engine.frequencies,
                pl_engine.localization_ratio,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("qk", "all"):
            _do(
                "qk",
                plotter.plot_qk_vs_penergy,
                pl_engine.frequencies,
                pl_engine.qks,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("hr_factor", "all"):
            _do(
                "hr_factor",
                plotter.plot_HR_factor_vs_penergy,
                pl_engine.frequencies,
                pl_engine.Sks,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("s_omega", "all"):
            _do(
                "s_omega",
                plotter.plot_S_omega_vs_penergy,
                pl_engine.frequencies,
                pl_engine.S_omega,
                pl_engine.omega_range,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("s_omega_sk", "all"):
            _do(
                "s_omega_sk",
                plotter.plot_S_omega_Sks_vs_penergy,
                pl_engine.frequencies,
                pl_engine.S_omega,
                pl_engine.omega_range,
                pl_engine.Sks,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("s_omega_locrat", "all"):
            _do(
                "s_omega_locrat",
                plotter.plot_S_omega_Sks_Loc_rat_vs_penergy,
                pl_engine.frequencies,
                pl_engine.S_omega,
                pl_engine.omega_range,
                pl_engine.Sks,
                pl_engine.localization_ratio,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("s_omega_ipr", "all"):
            _do(
                "s_omega_ipr",
                plotter.plot_S_omega_Sks_ipr_vs_penergy,
                pl_engine.frequencies,
                pl_engine.S_omega,
                pl_engine.omega_range,
                pl_engine.Sks,
                pl_engine.iprs,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("s_omega_ipr_alkauskas", "all"):
            _do(
                "s_omega_ipr_alkauskas",
                plotter.plot_S_omega_Sks_ipr_alkauskas_vs_penergy,
                pl_engine.frequencies,
                pl_engine.S_omega,
                pl_engine.omega_range,
                pl_engine.Sks,
                pl_engine.iprs_alkauskas,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("nk", "all"):
            _do(
                "nk",
                plotter.plot_nk_vs_penergy,
                pl_engine.frequencies,
                pl_engine.nks,
                pl_engine.temperature,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

        if pt in ("c_omega", "all"):
            _do(
                "c_omega",
                plotter.plot_C_omega_vs_penergy,
                pl_engine.frequencies,
                pl_engine.C_omega,
                pl_engine.omega_range,
                plot=False,
                out_dir=out_dir,
                max_freq=freq_limit,
                fig_format=fmt,
            )

        if pt in ("intensity", "all"):
            if not isinstance(pl_engine, Photoluminescence):
                raise ValueError(
                    "'intensity' plot type requires a Photoluminescence JSON. "
                    "For absorption use a Photoabsorption JSON with '-t absorption'."
                )
            _do(
                "intensity",
                plotter.plot_intensity_vs_penergy,
                pl_engine.frequencies,
                pl_engine.intensity,
                pl_engine.resolution,
                iplot_xlim,
                plot=False,
                out_dir=out_dir,
                iylim=parsed_iylim,
                fig_format=fmt,
            )

        if pt in ("absorption", "all"):
            if not isinstance(pl_engine, Photoabsorption):
                raise ValueError(
                    "'absorption' plot type requires a Photoabsorption JSON. "
                    "For PL emission use a Photoluminescence JSON with '-t intensity'."
                )
            _do(
                "absorption",
                plotter.plot_absorption_vs_penergy,
                frequencies=pl_engine.frequencies,
                absorption=pl_engine.absorption,
                resolution=pl_engine.resolution,
                xlim=iplot_xlim,
                plot=False,
                out_dir=out_dir,
                fig_format=fmt,
            )

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Failed to generate plot: {exc}")


# =====================================================================
# PL + ABSORPTION OVERLAY COMMAND
# =====================================================================


@main.command(name="overlay")
@click.option(
    "--pl",
    "pl_json",
    required=True,
    type=click.Path(exists=True),
    help="Path to a serialized Photoluminescence JSON file.",
)
@click.option(
    "--abs",
    "abs_json",
    required=True,
    type=click.Path(exists=True),
    help="Path to a serialized Photoabsorption JSON file.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output directory for the overlay figure.",
)
@click.option(
    "--fmt",
    default="pdf",
    help="Figure file format (e.g., pdf, png, svg).",
)
@click.option(
    "--iylim",
    default=None,
    help="Comma-separated y-axis limits for the overlay plot (e.g., '0,1.2').",
)
def overlay(pl_json, abs_json, out_dir, fmt, iylim):
    """Overlay PL emission and photoabsorption spectra on a single plot.

    Loads a Photoluminescence JSON (GS phonons) and a Photoabsorption JSON
    (ES phonons) and calls plot_pl_absorption_vs_penergy to render both on
    a shared energy axis.

    \b
    Example:
      defectpl overlay --pl pl.json --abs abs.json --out_dir figs/ --fmt png
    """
    from defectpl.defectpl import Photoluminescence, Photoabsorption
    from defectpl.plot import Plotter
    from monty.serialization import loadfn

    try:
        click.echo(f"Loading PL JSON: {pl_json}")
        pl_engine = loadfn(pl_json)
        if not isinstance(pl_engine, Photoluminescence):
            raise ValueError(
                f"'{pl_json}' does not contain a Photoluminescence object."
            )

        click.echo(f"Loading absorption JSON: {abs_json}")
        abs_engine = loadfn(abs_json)
        if not isinstance(abs_engine, Photoabsorption):
            raise ValueError(f"'{abs_json}' does not contain a Photoabsorption object.")

        plotter = Plotter()
        iplot_xlim = (
            max(0.0, min(pl_engine.EZPL, abs_engine.EZPL) - 2.0),
            max(pl_engine.EZPL, abs_engine.EZPL) + 1.0,
        )
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        plotter.plot_pl_absorption_vs_penergy(
            frequencies=pl_engine.frequencies,
            intensity=pl_engine.intensity,
            absorption=abs_engine.absorption,
            resolution=pl_engine.resolution,
            xlim=iplot_xlim,
            plot=False,
            out_dir=out_dir,
            fig_format=fmt,
        )
        click.echo(f"Overlay plot written to {out_dir}")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Failed to generate overlay plot: {exc}")


# =====================================================================
# CONFIGURATION COORDINATE DISPLACEMENT (DQ) COMMANDS
# =====================================================================


@main.command(name="dq")
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    default=None,
    help="Optional destination path tracker targeting an extracted JSON file.",
)
@click.option(
    "--format",
    "-f",
    "out_format",
    type=click.Choice(["plain", "json"], case_sensitive=False),
    default="plain",
    help="Controls format configuration mapping for structural output logs.",
)
def dq(structure1, structure2, out_path, out_format):
    """Calculate mass-weighted generalized configuration coordinate displacement delta Q."""
    from pymatgen.core import Structure
    from defectpl.io.vasp import calc_delta_Q

    try:
        s1 = Structure.from_file(structure1)
        s2 = Structure.from_file(structure2)
        delta_q = calc_delta_Q(s1, s2)
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Failed to calculate mass-weighted deltaQ: {exc}")

    output_payload = {
        "structure1": str(structure1),
        "structure2": str(structure2),
        "deltaQ": delta_q,
    }

    if out_format.lower() == "plain":
        click.echo(f"{delta_q:.8f}")
    else:
        click.echo(json.dumps(output_payload, indent=2))

    if out_path:
        p = Path(out_path)
        p.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        click.echo(f"Wrote configuration coordinate delta Q records to {p}")


@main.command(name="spectra1d")
@click.option(
    "--ezpl", required=True, type=float, help="Zero-phonon line energy value (eV)."
)
@click.option(
    "--w1",
    required=True,
    type=float,
    help="Vibrational frequency tracking excited state wells (meV).",
)
@click.option(
    "--w2",
    required=True,
    type=float,
    help="Vibrational frequency tracking ground state wells (meV).",
)
@click.option(
    "--dq_val", required=True, type=float, help="Generalized coordinate offset delta Q."
)
@click.option(
    "--temp",
    default=300.0,
    type=float,
    help="System operational temperature parameter in Kelvin.",
)
@click.option(
    "--e0",
    default=0.0,
    type=float,
    help="Energy evaluation grid starting floor parameter (eV).",
)
@click.option(
    "--de",
    default=0.001,
    type=float,
    help="Grid channel sampling increment step dimension parameter (eV).",
)
@click.option(
    "--points",
    default=5000,
    type=int,
    help="Total integrated point matrix rows arrays dimension count.",
)
@click.option(
    "--nn1",
    default=22,
    type=int,
    help="Maximum quantum state boundary index cut for excited wells.",
)
@click.option(
    "--nn2",
    default=52,
    type=int,
    help="Maximum quantum state boundary index cut for ground wells.",
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Launches visualizer mapping normalized shapes.",
)
@click.option(
    "--save_prefix",
    default="vibrational_1d",
    help="Prefix for files output during metrics serialization.",
)
def spectra1d(ezpl, w1, w2, dq_val, temp, e0, de, points, nn1, nn2, plot, save_prefix):
    """Run decoupled analytical 1D Franck-Condon harmonic approximation lineshape tracks."""
    from defectpl.defectpl import VibrationalSpectra1D

    try:
        spec = VibrationalSpectra1D(
            EZPL=ezpl,
            w1_meV=w1,
            w2_meV=w2,
            DQ=dq_val,
            T=temp,
            E0=e0,
            dE=de,
            M=points,
            NN1=nn1,
            NN2=nn2,
        )
        click.echo("Computing transition metrics under overlap matrices rules...")
        spec.compute_lineshape()
        spec.get_peak_position()
        spec.get_fwhm()

        spec.save_results(
            overlap_file=f"{save_prefix}_overlap.json",
            lineshape_file=f"{save_prefix}_lineshape.json",
        )
        if plot:
            spec.plot_lineshape(save_file=f"{save_prefix}_plot.pdf")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Analytical processing execution failed: {exc}")


# =====================================================================
# CONFIGURATION COORDINATE DIAGRAM (CCD) PIPELINES
# =====================================================================


@main.command(name="setup-ccd")
@click.option(
    "--gs",
    required=True,
    type=click.Path(exists=True),
    help="Reference ground structure configuration.",
)
@click.option(
    "--es",
    required=True,
    type=click.Path(exists=True),
    help="Reference excited structure configuration.",
)
@click.option(
    "--out_dir",
    default="./ccd_calculations",
    help="Root directory endpoint mapping out workspace calculations.",
)
@click.option(
    "--tmpl_gs",
    required=True,
    type=click.Path(exists=True),
    help="Directory hosting ground VASP parameter scripts.",
)
@click.option(
    "--tmpl_es",
    required=True,
    type=click.Path(exists=True),
    help="Directory hosting excited VASP parameter scripts.",
)
@click.option(
    "--steps",
    default="-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2",
    help="Comma-separated fraction list tracing interpolation paths.",
)
def setup_ccd(gs, es, out_dir, tmpl_gs, tmpl_es, steps):
    """Generate linear interpolation structure configuration spaces for automated VASP execution parameters."""
    from pymatgen.core import Structure
    from defectpl.defectpl import ConfigurationCoordinateDiagram

    try:
        s_gs = Structure.from_file(gs)
        s_es = Structure.from_file(es)
        displacements = [float(x.strip()) for x in steps.split(",")]

        ccd = ConfigurationCoordinateDiagram(ground_struct=s_gs, excited_struct=s_es)
        ccd.setup_calculations(
            displacements=displacements,
            output_dir=out_dir,
            ground_input_dir=tmpl_gs,
            excited_input_dir=tmpl_es,
        )
        click.echo(
            f"Interpolated task configuration structures tree setup complete at: {out_dir}"
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Calculations generation workflow initialization failed: {exc}"
        )


@main.command(name="analyze-ccd")
@click.option(
    "--gs",
    required=True,
    type=click.Path(exists=True),
    help="Baseline structural geometry configuration (GS).",
)
@click.option(
    "--es",
    required=True,
    type=click.Path(exists=True),
    help="Baseline structural geometry configuration (ES).",
)
@click.option(
    "--gs_runs",
    required=True,
    help="Space-separated paths targeting completed ground state xml run parameters.",
)
@click.option(
    "--es_runs",
    required=True,
    help="Space-separated paths targeting completed excited state xml run parameters.",
)
@click.option(
    "--de",
    default=0.0,
    type=float,
    help="Energy gap minimum offset scalar separation factor (eV).",
)
@click.option(
    "--save_plot",
    default=None,
    help="If provided, exports the diagram visualization path layout safely.",
)
def analyze_ccd(gs, es, gs_runs, es_runs, de, save_plot):
    """Fit calculated Potential Energy Surfaces data arrays, extract well parameters, and report metrics."""
    from pymatgen.core import Structure
    from defectpl.defectpl import ConfigurationCoordinateDiagram

    try:
        s_gs = Structure.from_file(gs)
        s_es = Structure.from_file(es)

        paths_gs = [p.strip() for p in gs_runs.split(" ") if p.strip()]
        paths_es = [p.strip() for p in es_runs.split(" ") if p.strip()]

        ccd = ConfigurationCoordinateDiagram(ground_struct=s_gs, excited_struct=s_es)
        w_g, w_e = ccd.analyze_ccd(
            ground_vaspruns=paths_gs,
            excited_vaspruns=paths_es,
            dE=de,
            save_plot=save_plot,
        )
        click.echo("\nFitted Harmonic Well Parameters Found:")
        click.echo(f"Ground state effective phonon frequency energy: {w_g:.4f} eV")
        click.echo(f"Excited state effective phonon frequency energy: {w_e:.4f} eV")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Failed to fit potential energy curvature points: {exc}"
        )


# =====================================================================
# DATA BENCHMARKING AND UTILITY METRICS COMMANDS
# =====================================================================


@main.command(name="compare-json")
@click.option(
    "--files",
    required=True,
    help="Space-separated file lists targeting processed property JSON files.",
)
@click.option(
    "--xmin",
    default=None,
    type=float,
    help="Window boundary minimum cutoff limit for photon energy range.",
)
@click.option(
    "--xmax",
    default=None,
    type=float,
    help="Window boundary maximum cutoff limit for photon energy range.",
)
@click.option(
    "--legends",
    default=None,
    help="Comma-separated labels mapping structural targets sequentially.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Directory destination track location to drop compared figures.",
)
@click.option(
    "--fmt", default="pdf", help="Image output compression extension style layout."
)
def compare_json(files, xmin, xmax, legends, out_dir, fmt):
    """Generate comparative visualization graphs detailing differences across serialized static property JSON files."""
    from defectpl.plot import comparepl

    try:
        target_files = [Path(f.strip()) for f in files.split(" ") if f.strip()]
        labels = [lbl.strip() for lbl in legends.split(",")] if legends else None
        xlim = (xmin, xmax) if (xmin is not None and xmax is not None) else None

        comparepl(
            properties_files=target_files,
            xlim=xlim,
            legends=labels,
            out_dir=out_dir,
            fig_format=fmt,
        )
        click.echo("Comparative property spectrum graph compiled successfully.")
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Comparison array aggregation processing failure encountered: {exc}"
        )


@main.command(name="compare-yaml")
@click.option(
    "--yamls",
    required=True,
    help="Space-separated file paths listing phonopy band.yaml configuration inputs.",
)
@click.option(
    "--gs",
    required=True,
    type=click.Path(exists=True),
    help="Ground state structure template (CONTCAR).",
)
@click.option(
    "--es",
    required=True,
    type=click.Path(exists=True),
    help="Excited state structure template (CONTCAR).",
)
@click.option(
    "--out_dir",
    default="./",
    help="Workspace path to save comparisons charts profile graphs.",
)
@click.option(
    "--ezpl", default=1.95, type=float, help="Zero-phonon line transition point (eV)."
)
@click.option("--gamma", default=2.0, type=float, help="Broadening parameter factor.")
@click.option(
    "--xmin", default=1.0, type=float, help="Minimum energy domain layout bound limit."
)
@click.option(
    "--xmax", default=3.0, type=float, help="Maximum energy domain layout bound limit."
)
@click.option(
    "--file_name",
    default="compare_yaml_intensity.pdf",
    help="Output plot filename template destination.",
)
def compare_yaml(yamls, gs, es, out_dir, ezpl, gamma, xmin, xmax, file_name):
    """Dynamically compile, build, and plot comparative intensities for lists of phonopy inputs configurations."""
    from pymatgen.core import Structure
    from defectpl.io.vasp import run_dynamic_yaml_comparison

    try:
        band_yaml_files = [Path(f.strip()) for f in yamls.split(" ") if f.strip()]
        struct_gs = Structure.from_file(gs)
        struct_es = Structure.from_file(es)

        out_path = run_dynamic_yaml_comparison(
            band_yaml_files=band_yaml_files,
            gs_structure=struct_gs,
            es_structure=struct_es,
            out_dir=out_dir,
            ezpl=ezpl,
            gamma=gamma,
            xmin=xmin,
            xmax=xmax,
            file_name=file_name,
        )
        click.echo(
            f"Dynamic execution spectra comparison chart saved successfully to {out_path}."
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Dynamic multi-yaml calculation task execution dropped: {exc}"
        )


# =====================================================================
# PHONON ANALYSIS COMMANDS (eV Metrics Scale)
# =====================================================================


@main.command(name="phonon-fc")
@click.argument("vasrun_xml", type=click.Path(exists=True))
@click.option(
    "--hdf5",
    is_flag=True,
    default=False,
    help="Write output force constants using HDF5 format structures.",
)
@click.option(
    "--log_level", default=1, type=int, help="Verbosity log tracking scale values."
)
def phonon_fc(vasrun_xml, hdf5, log_level):
    """Parse a vasprun.xml file and write extracted force constants to a FORCE_CONSTANTS file."""
    from defectpl.phonon import create_force_constants_from_vasprun

    status = create_force_constants_from_vasprun(
        vasrun_xml, is_hdf5=hdf5, log_level=log_level
    )
    if status == 0:
        click.echo("Force constants written successfully.")
    else:
        raise click.ClickException(
            "Phonopy internal processor failed to write force constants matrix."
        )


@main.command(name="phonon-symm")
@click.option(
    "--poscar",
    default="./POSCAR",
    help="Target VASP structural base unit cell geometry filepath.",
)
@click.option(
    "--fc",
    default=None,
    help="Optional destination path tracker targeting precalculated FORCE_CONSTANTS.",
)
@click.option(
    "--fs",
    default=None,
    help="Optional destination path tracker targeting precalculated FORCE_SETS files.",
)
@click.option(
    "--dim",
    default="1 1 1",
    help="Space-separated scaling factors defining supercell dimensions.",
)
@click.option(
    "--symprec",
    default=1e-5,
    type=float,
    help="Symmetry recognition grid matching tolerance constraints.",
)
def phonon_symm(poscar, fc, fs, dim, symprec):
    """Evaluate point-group irreducible representations (irreps) metrics at the Gamma point."""
    from defectpl.phonon import calculate_phonon_symmetries

    if not fc and not fs:
        raise click.UsageError(
            "Execution halted: Provide at least one valid data file source tracking --fc or --fs data entries."
        )

    try:
        calculate_phonon_symmetries(
            unitcell_path=poscar,
            force_constants_path=fc,
            force_sets_path=fs,
            dimension=dim,
            symprec=symprec,
        )

        click.echo("\nAll the IRs are saved into irreps.yaml file.")

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Symmetry parser pipeline execution failed: {exc}")


@main.command(name="phonon-band")
@click.option(
    "--poscar",
    default="./POSCAR",
    help="Target VASP structural base unit cell geometry filepath.",
)
@click.option(
    "--fc",
    default="./FORCE_CONSTANTS",
    help="Source force constants tracking parameter file matrix.",
)
@click.option(
    "--dim",
    default="1 1 1",
    help="Space-separated scaling factors defining supercell dimensions.",
)
@click.option(
    "--out",
    default="band.yaml",
    help="Destination file path where the compiled Phonopy YAML structure will be dropped.",
)
def phonon_band(poscar, fc, dim, out):
    """Evaluate phonon properties at the Gamma point from a FORCE_CONSTANTS file and write to a band.yaml file."""
    from defectpl.phonon import calculate_gamma_phonon_to_band_yaml

    try:
        calculate_gamma_phonon_to_band_yaml(
            unitcell_filename=poscar,
            force_constants_filename=fc,
            dimension=dim,
            output_filename=out,
        )
        click.echo(
            f"Phonon path calculations complete. Records dumped safely to: {out}"
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Failed to compile single-point band structure output: {exc}"
        )


@main.command(name="phonon-parse")
@click.argument("band_yaml", type=click.Path(exists=True))
@click.option(
    "--json_out",
    default="gamma_phonon_properties.json",
    help="Destination output filename tracker to drop JSON dump data records.",
)
def phonon_parse(band_yaml, json_out):
    """Parse a band.yaml file to convert all frequencies to eV and serialize into an MSONable JSON data model."""
    from defectpl.phonon import extract_gamma_phonon_data

    try:
        phonon_model = extract_gamma_phonon_data(band_yaml)
        p = Path(json_out)
        p.write_text(phonon_model.to_json(), encoding="utf-8")

        click.echo(
            f"Successfully processed {phonon_model.nmodes} modes tracking {phonon_model.natoms} atoms."
        )
        click.echo(
            f"Energetic values converted to eV and database profile written to: {json_out}"
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Parsing engine dropped properties evaluation tasks: {exc}"
        )


# =====================================================================
# KOHN-SHAM LEVEL VISUALIZATION COMMANDS
# =====================================================================


@main.command(name="ksplot")
@click.argument("eigenval", type=click.Path(exists=True))
@click.option(
    "--vbm",
    required=True,
    type=float,
    help="Energy value of the Valence Band Maximum (eV).",
)
@click.option(
    "--cbm",
    required=True,
    type=float,
    help="Energy value of the Conduction Band Minimum (eV).",
)
@click.option(
    "--espan",
    default=1.0,
    type=float,
    help="Energy canvas buffer padding depth beyond band edges (eV).",
)
@click.option(
    "--kidx",
    default=0,
    type=int,
    help="Target sequential k-point list array index index grid target.",
)
@click.option(
    "--out_img",
    default="ks_plot.png",
    help="File layout path target endpoint to drop final generated graphic.",
)
@click.option(
    "--out_json",
    default=None,
    help="Optional destination to export serialized MSONable JSON data model records.",
)
def ksplot(eigenval, vbm, cbm, espan, kidx, out_img, out_json):
    """
    Extract, resolve degeneracies, and plot spin-polarized Kohn-Sham electronic states near the bandgap.

    Usage:\n
      defectpl ksplot ./EIGENVAL --vbm 9.6747 --cbm 13.7934 --kidx 0 --out_img gap_levels.png
    """
    from defectpl.io.vasp import run_kohn_sham_analysis

    try:
        click.echo(
            f"Initializing parsing matrix structures tracking target point: {eigenval}"
        )

        run_kohn_sham_analysis(
            eigenval_path=eigenval,
            vbm=vbm,
            cbm=cbm,
            espan=espan,
            k_idx=kidx,
            output_img=out_img,
            output_json=out_json,
        )

        click.echo(
            f"Successfully generated Kohn-Sham electronic levels matrix diagram at: {out_img}"
        )
        if out_json:
            click.echo(
                f"Exported MSONable state database models JSON records to: {out_json}"
            )

    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(
            f"Kohn-Sham calculation plotting tracking layer crashed: {exc}"
        )


# Link the nested PL group structure into our root application workspace
main.add_command(pl_group)

# Link the absorption group into the root application workspace
main.add_command(absorption_group)


# =====================================================================
# PARTICIPATION RATIO COMMANDS (pr group)
# =====================================================================


@click.group(name="pr")
def pr_group():
    """
    Electronic-state Participation Ratio (P-ratio) and IPR tools.

    Computes how much of each Kohn-Sham wavefunction is localised on the
    defect neighbourhood, using PROCAR + defect_entry.json as input.

    Subcommands:

    \b
      calc       Run the full P-ratio / IPR calculation.
      batch      Batch-process charge-state subdirectories.
      summary    Pretty-print an existing participation_ratio.json.
      top        List the N most-localised states.
      make-entry Generate defect_entry.json without pydefect.
      make-dsi   Generate defect_structure_info.json without pydefect.
    """


# ------- shared option factories ----------------------------------------


def _opt_procar(f):
    return click.option(
        "--procar",
        "-p",
        default="PROCAR",
        show_default=True,
        type=click.Path(dir_okay=False),
        help="Path to VASP PROCAR file.  Needs LORBIT=11 or 12 in INCAR.",
    )(f)


def _opt_entry(f):
    return click.option(
        "--entry",
        "-e",
        default="defect_entry.json",
        show_default=True,
        type=click.Path(dir_okay=False),
        help="Path to defect_entry.json (from 'defectpl pr make-entry' or pydefect).",
    )(f)


def _opt_dsi(f):
    return click.option(
        "--dsi",
        "-s",
        default=None,
        type=click.Path(dir_okay=False),
        help=(
            "Path to defect_structure_info.json.  When provided, neighbour "
            "atom indices are read directly (recommended).  "
            "Falls back to distance-based search when absent."
        ),
    )(f)


def _opt_poscar_pr(f):
    return click.option(
        "--poscar",
        default=None,
        type=click.Path(dir_okay=False),
        help=(
            "Path to POSCAR or CONTCAR for the distance-based fallback "
            "neighbour search.  Auto-detected (CONTCAR > POSCAR) when omitted."
        ),
    )(f)


def _opt_cutoff(f):
    return click.option(
        "--cutoff",
        "-c",
        default=3.5,
        show_default=True,
        type=float,
        help="Neighbour cut-off radius in Å (fallback distance search only).",
    )(f)


def _opt_out_dir(f):
    return click.option(
        "--out",
        "-o",
        default=".",
        show_default=True,
        type=click.Path(file_okay=False),
        help="Output directory for participation_ratio.json and .csv.",
    )(f)


# ------- pretty-print helper --------------------------------------------


def _print_pr_summary(result: dict, top_n: int = 15) -> None:
    sep = "─" * 76
    click.echo(f"\n{sep}")
    click.echo(f"  Defect  : {result['defect_name']}")
    cx, cy, cz = result["defect_center"]
    click.echo(f"  Centre  : ({cx:.4f}, {cy:.4f}, {cz:.4f})  [fractional]")
    click.echo(
        f"  Neighbours ({len(result['neighbor_atom_indices'])} atoms): "
        f"{result['neighbor_atom_indices']}"
    )
    click.echo(
        f"  Grid    : {result['n_spins']} spin(s) × "
        f"{result['n_kpoints']} k-pt(s) × {result['n_bands']} band(s)  "
        f"| {result['n_atoms']} ions"
    )
    click.echo(f"{sep}")

    rows = []
    for sp_label, kpt_dict in result["data"].items():
        for kpt_label, band_dict in kpt_dict.items():
            kpt_idx = int(kpt_label.split("_")[1])
            for band_label, vals in band_dict.items():
                band_idx = int(band_label.split("_")[1])
                rows.append(
                    (
                        sp_label,
                        kpt_idx,
                        band_idx,
                        vals.get("energy"),
                        vals.get("occupation"),
                        vals["p_ratio"],
                        vals["ipr"],
                    )
                )
    rows.sort(key=lambda x: -x[5])

    click.echo(
        f"  {'Spin':<7} {'Kpt':>4} {'Band':>5}  "
        f"{'Energy(eV)':>11}  {'Occ':>5}  "
        f"{'P-ratio':>8}  {'IPR':>10}"
    )
    click.echo(
        f"  {'─' * 7} {'─' * 4} {'─' * 5}  {'─' * 11}  {'─' * 5}  {'─' * 8}  {'─' * 10}"
    )
    for sp, ik, ib, en, occ, pr, ipr in rows[:top_n]:
        en_str = f"{en:11.4f}" if en is not None else f"{'N/A':>11}"
        occ_str = f"{occ:5.3f}" if occ is not None else f"{'N/A':>5}"
        click.echo(
            f"  {sp:<7} {ik:>4} {ib:>5}  {en_str}  {occ_str}  {pr:8.4f}  {ipr:10.6f}"
        )
    click.echo(f"{sep}\n")


# ------- defectpl pr calc -----------------------------------------------


@pr_group.command("calc")
@_opt_procar
@_opt_entry
@_opt_dsi
@_opt_poscar_pr
@_opt_cutoff
@_opt_out_dir
@click.option(
    "--top",
    "top_n",
    default=15,
    show_default=True,
    type=int,
    help="Number of most-localised states to print in the terminal summary.",
)
@click.option(
    "--no-csv",
    is_flag=True,
    default=False,
    help="Skip writing the flat CSV summary file.",
)
@click.option(
    "--native-procar",
    is_flag=True,
    default=False,
    help="Force the built-in PROCAR parser instead of pymatgen's.",
)
def pr_calc(procar, entry, dsi, poscar, cutoff, out, top_n, no_csv, native_procar):
    """
    Calculate P-ratio and IPR for every (spin, k-point, band).

    Reads PROCAR + defect_entry.json (and optionally
    defect_structure_info.json) for the defect neighbour list.

    Output files written to --out directory:

    \b
      participation_ratio.json         – full nested results
      participation_ratio_summary.csv  – flat table (unless --no-csv)
    """
    import logging as _logging
    from defectpl.participation_ratio import ParticipationRatioCalculator

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    calc = ParticipationRatioCalculator(
        procar=procar,
        defect_entry=entry,
        defect_structure_info=dsi,
        poscar=poscar,
        cutoff_radius=cutoff,
        use_pymatgen=not native_procar,
    )
    try:
        result = calc.run()
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))

    out_dir = Path(out)
    calc.to_json(out_dir / "participation_ratio.json")
    if not no_csv:
        calc.to_csv(out_dir / "participation_ratio_summary.csv")

    _print_pr_summary(result, top_n=top_n)
    click.secho("Done.", fg="green", bold=True)


# ------- defectpl pr batch ----------------------------------------------


@pr_group.command("batch")
@click.option(
    "--dir",
    "-d",
    "batch_dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, exists=True),
    help="Parent directory to scan for defect charge-state sub-directories.",
)
@_opt_cutoff
@click.option(
    "--no-csv",
    is_flag=True,
    default=False,
    help="Skip writing per-directory CSV files.",
)
@click.option(
    "--combined-csv",
    "combined_csv",
    default="batch_participation_ratio.csv",
    show_default=True,
    help="Name of the combined batch CSV written in --dir.",
)
@click.option(
    "--native-procar",
    is_flag=True,
    default=False,
    help="Force the built-in PROCAR parser instead of pymatgen's.",
)
def pr_batch(batch_dir, cutoff, no_csv, combined_csv, native_procar):
    """
    Run 'pr calc' for every subdirectory that contains PROCAR + defect_entry.json.

    A combined CSV of all results is written in --dir.
    """
    import csv as _csv
    import logging as _logging
    from defectpl.participation_ratio import ParticipationRatioCalculator

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    batch_path = Path(batch_dir)

    dirs = sorted(d for d in batch_path.iterdir() if d.is_dir())
    if not dirs:
        click.echo(f"No sub-directories found in {batch_path}.")
        return

    all_rows = []
    n_ok = 0

    for d in dirs:
        procar_p = d / "PROCAR"
        entry_p = d / "defect_entry.json"
        if not procar_p.exists() or not entry_p.exists():
            continue

        click.echo(f"\n>>> {d.name}")
        dsi_p = d / "defect_structure_info.json"
        poscar_p = next(
            (d / n for n in ("CONTCAR", "POSCAR") if (d / n).exists()), None
        )

        try:
            calc = ParticipationRatioCalculator(
                procar=procar_p,
                defect_entry=entry_p,
                defect_structure_info=dsi_p if dsi_p.exists() else None,
                poscar=poscar_p,
                cutoff_radius=cutoff,
                use_pymatgen=not native_procar,
            )
            result = calc.run()
            calc.to_json(d / "participation_ratio.json")
            if not no_csv:
                calc.to_csv(d / "participation_ratio_summary.csv")
            n_ok += 1

            for sp_label, kpt_dict in result["data"].items():
                for kpt_label, band_dict in kpt_dict.items():
                    kpt_idx = int(kpt_label.split("_")[1])
                    for band_label, vals in band_dict.items():
                        band_idx = int(band_label.split("_")[1])
                        all_rows.append(
                            dict(
                                defect=result["defect_name"],
                                dir=d.name,
                                spin=sp_label,
                                kpt=kpt_idx,
                                band=band_idx,
                                energy=vals.get("energy"),
                                occ=vals.get("occupation"),
                                p_ratio=vals["p_ratio"],
                                ipr=vals["ipr"],
                            )
                        )
        except Exception as exc:
            if _verbose:
                click.secho(f"  ERROR in {d.name}:", fg="red", err=True)
                click.echo(traceback.format_exc(), err=True)
            else:
                click.secho(f"  SKIPPED ({d.name}): {exc}", fg="yellow")

    if all_rows:
        csv_path = batch_path / combined_csv
        fieldnames = [
            "defect",
            "dir",
            "spin",
            "kpt",
            "band",
            "energy",
            "occ",
            "p_ratio",
            "ipr",
        ]
        with open(csv_path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        click.echo(f"\n[Batch CSV] {csv_path}  ({len(all_rows)} rows)")

    click.secho(
        f"\nBatch complete: {n_ok}/{len(dirs)} directories processed.",
        fg="green",
        bold=True,
    )


# ------- defectpl pr summary --------------------------------------------


@pr_group.command("summary")
@click.argument(
    "json_file",
    default="participation_ratio.json",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option(
    "--top",
    "top_n",
    default=15,
    show_default=True,
    type=int,
    help="Number of most-localised states to print.",
)
def pr_summary(json_file, top_n):
    """
    Pretty-print a participation_ratio.json file (no recalculation).

    JSON_FILE  path to participation_ratio.json  [default: ./participation_ratio.json]
    """
    with open(json_file) as fh:
        result = json.load(fh)
    _print_pr_summary(result, top_n=top_n)


# ------- defectpl pr top ------------------------------------------------


@pr_group.command("top")
@click.argument(
    "json_file",
    default="participation_ratio.json",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option(
    "--n",
    "top_n",
    default=10,
    show_default=True,
    type=int,
    help="Number of states to list.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(["p_ratio", "ipr"]),
    default="p_ratio",
    show_default=True,
    help="Metric used for ranking.",
)
def pr_top(json_file, top_n, metric):
    """
    List the N most-localised states sorted by P-ratio or IPR.

    JSON_FILE  path to participation_ratio.json
    """
    from defectpl.participation_ratio import ParticipationRatioCalculator

    with open(json_file) as fh:
        result = json.load(fh)

    calc = ParticipationRatioCalculator.__new__(ParticipationRatioCalculator)
    calc._result = result  # type: ignore[attr-defined]

    top = calc.top_localized(n=top_n, metric=metric)

    click.echo(f"\nTop {top_n} states by {metric}  ({result['defect_name']})")
    click.echo("─" * 68)
    click.echo(
        f"  {'Spin':<7} {'Kpt':>4} {'Band':>5}  "
        f"{'Energy(eV)':>11}  {'P-ratio':>8}  {'IPR':>10}"
    )
    click.echo(f"  {'─' * 7} {'─' * 4} {'─' * 5}  {'─' * 11}  {'─' * 8}  {'─' * 10}")
    for row in top:
        en_str = (
            f"{row['energy']:11.4f}" if row["energy"] is not None else f"{'N/A':>11}"
        )
        click.echo(
            f"  {row['spin']:<7} {row['kpt']:>4} {row['band']:>5}  "
            f"{en_str}  {row['p_ratio']:8.4f}  {row['ipr']:10.6f}"
        )
    click.echo()


# ------- defectpl pr make-entry -----------------------------------------


@pr_group.command("make-entry")
@click.option(
    "--name",
    "-n",
    required=True,
    help="Defect label, e.g. 'Va_O1_2' for oxygen vacancy charge +2.",
)
@click.option(
    "--center",
    "-c",
    default=None,
    help=(
        "Fractional coordinates of the defect centre as 'x,y,z' or 'x y z'.  "
        "Required when --perfect / --defect are not given."
    ),
)
@click.option(
    "--perfect",
    "-P",
    default=None,
    type=click.Path(dir_okay=False),
    help="POSCAR/CONTCAR of the perfect (undoped) supercell for auto-detection.",
)
@click.option(
    "--defect",
    "-D",
    default=None,
    type=click.Path(dir_okay=False),
    help="POSCAR/CONTCAR of the defect supercell for auto-detection.",
)
@click.option(
    "--site-tol",
    default=0.5,
    show_default=True,
    type=float,
    help="Cartesian distance tolerance (Å) for site matching in auto-detection.",
)
@click.option(
    "--out",
    "-o",
    default="defect_entry.json",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Output path for defect_entry.json.",
)
def pr_make_entry(name, center, perfect, defect, site_tol, out):
    """
    Generate defect_entry.json without pydefect.

    Two modes:

    \b
    Manual   -- provide --center as fractional coordinates:
                defectpl pr make-entry --name Va_O1_2 --center 0.5,0.5,0.5

    Auto     -- provide perfect + defect POSCAR/CONTCAR (vacancy auto-detected):
                defectpl pr make-entry --name Va_O1_2 \\
                    --perfect POSCAR_perfect --defect CONTCAR
    """
    from defectpl.defect_utils import make_defect_entry, parse_frac_coords

    try:
        center_coords = parse_frac_coords(center) if center else None
        payload = make_defect_entry(
            name=name,
            center=center_coords,
            perfect_poscar=perfect,
            defect_poscar=defect,
            out_path=out,
            site_tol=site_tol,
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))

    click.echo(f"Defect name   : {payload['name']}")
    cx, cy, cz = payload["defect_center"]
    click.echo(f"Defect centre : ({cx:.6f}, {cy:.6f}, {cz:.6f})  [fractional]")
    click.echo(f"Defect type   : {payload.get('defect_type', 'manual')}")
    click.secho(f"Written       : {out}", fg="green", bold=True)


# ------- defectpl pr make-dsi -------------------------------------------


@pr_group.command("make-dsi")
@click.option(
    "--poscar",
    "-p",
    required=True,
    type=click.Path(dir_okay=False, exists=True),
    help="POSCAR/CONTCAR of the defect supercell.",
)
@click.option(
    "--center",
    "-c",
    required=True,
    help="Fractional coordinates of the defect centre as 'x,y,z' or 'x y z'.",
)
@click.option(
    "--cutoff",
    "-r",
    default=3.5,
    show_default=True,
    type=float,
    help="Neighbour search radius in Å.",
)
@click.option(
    "--out",
    "-o",
    default="defect_structure_info.json",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Output path for defect_structure_info.json.",
)
def pr_make_dsi(poscar, center, cutoff, out):
    """
    Generate defect_structure_info.json without pydefect.

    Finds all atoms within --cutoff Å of the defect centre and writes their
    indices to defect_structure_info.json, which is used by 'pr calc'.

    Example:

    \b
        defectpl pr make-dsi --poscar CONTCAR --center 0.5,0.5,0.5 --cutoff 3.5
    """
    from defectpl.defect_utils import make_defect_structure_info, parse_frac_coords

    try:
        center_coords = parse_frac_coords(center)
        payload = make_defect_structure_info(
            poscar=poscar,
            defect_center_frac=center_coords,
            cutoff_radius=cutoff,
            out_path=out,
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))

    click.echo(f"Neighbours found : {payload['n_neighbors']}")
    click.echo(f"Indices          : {payload['neighbor_atom_indices']}")
    click.secho(f"Written          : {out}", fg="green", bold=True)


@pr_group.command("plot")
@click.argument(
    "json_file",
    default="participation_ratio.json",
    metavar="JSON_FILE",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--xaxis",
    "-x",
    type=click.Choice(["energy", "band"], case_sensitive=False),
    default="energy",
    show_default=True,
    help="X-axis quantity: energy (eV) or band index.",
)
@click.option(
    "--metric",
    "-m",
    default="p_ratio",
    type=click.Choice(["p_ratio", "ipr"], case_sensitive=False),
    show_default=True,
    help="Y-axis metric.",
)
@click.option(
    "--threshold",
    "-t",
    default=0.2,
    show_default=True,
    help="Horizontal threshold line value.",
)
@click.option(
    "--vbm",
    default=None,
    type=float,
    help="VBM energy (eV) — drawn as vertical line on energy plot.",
)
@click.option(
    "--cbm",
    default=None,
    type=float,
    help="CBM energy (eV) — drawn as vertical line on energy plot.",
)
@click.option("--emin", default=None, type=float, help="Lower energy filter (eV).")
@click.option("--emax", default=None, type=float, help="Upper energy filter (eV).")
@click.option(
    "--kpt",
    "kpt_idx",
    default=0,
    show_default=True,
    type=int,
    help="0-based k-point index to plot.",
)
@click.option(
    "--out",
    "-o",
    default=None,
    help="Output image file (default: pr_energy.png or pr_band.png).",
)
@click.option("--title", default=None, help="Plot title (defaults to defect name).")
def pr_plot(
    json_file, xaxis, metric, threshold, vbm, cbm, emin, emax, kpt_idx, out, title
):
    """Scatter plot of P-ratio or IPR versus energy or band index.

    \b
    Filled markers = occupied (occ ≥ 0.5); open markers = empty.
    Spin channels are colour-coded: blue = spin ↑, red = spin ↓.

    \b
        # P-ratio vs energy (default)
        defectpl pr plot participation_ratio.json

        # P-ratio vs band index
        defectpl pr plot participation_ratio.json --xaxis band

        # IPR vs energy with band-gap markers
        defectpl pr plot participation_ratio.json --metric ipr --vbm 5.2 --cbm 8.1
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        raise click.ClickException(
            "matplotlib is required for 'pr plot'.  Install with: pip install matplotlib"
        )

    from defectpl.participation_ratio import plot_pr_vs_energy, plot_pr_vs_band_index

    with open(json_file) as fh:
        result = json.load(fh)

    out_path = out or (f"pr_{xaxis}.png")

    try:
        if xaxis == "energy":
            plot_pr_vs_energy(
                result,
                metric=metric,
                threshold=threshold,
                vbm=vbm,
                cbm=cbm,
                emin=emin,
                emax=emax,
                kpt_idx=kpt_idx,
                title=title,
                out=out_path,
            )
        else:
            plot_pr_vs_band_index(
                result,
                metric=metric,
                threshold=threshold,
                emin=emin,
                emax=emax,
                kpt_idx=kpt_idx,
                title=title,
                out=out_path,
            )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))

    click.secho(f"Saved: {out_path}", fg="green", bold=True)


@pr_group.command("ksplot")
@click.option(
    "--eigenval",
    "-e",
    default="EIGENVAL",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Path to VASP EIGENVAL file.",
)
@click.option(
    "--pr-json",
    "pr_json",
    default="participation_ratio.json",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="participation_ratio.json from 'defectpl pr calc'.",
)
@click.option(
    "--vbm", required=True, type=float, help="Valence Band Maximum energy in eV."
)
@click.option(
    "--cbm", required=True, type=float, help="Conduction Band Minimum energy in eV."
)
@click.option(
    "--espan",
    default=1.0,
    show_default=True,
    type=float,
    help="Energy padding above/below VBM/CBM in eV.",
)
@click.option(
    "--metric",
    "-m",
    default="p_ratio",
    type=click.Choice(["p_ratio", "ipr"], case_sensitive=False),
    show_default=True,
    help="Localization metric for colour coding.",
)
@click.option(
    "--cmap", default="RdYlGn_r", show_default=True, help="Matplotlib colormap name."
)
@click.option(
    "--vmin", default=0.0, show_default=True, type=float, help="Colormap lower bound."
)
@click.option(
    "--vmax", default=1.0, show_default=True, type=float, help="Colormap upper bound."
)
@click.option(
    "--kidx",
    "kpt_idx",
    default=0,
    show_default=True,
    type=int,
    help="0-based k-point index (for multi-k EIGENVAL).",
)
@click.option(
    "--out",
    "-o",
    default="ks_pr_plot.png",
    show_default=True,
    help="Output image file (png/pdf/svg).",
)
@click.option("--title", default=None, help="Plot title.")
def pr_ksplot(
    eigenval, pr_json, vbm, cbm, espan, metric, cmap, vmin, vmax, kpt_idx, out, title
):
    """KS level diagram with levels colour-coded by P-ratio or IPR.

    Reads a VASP EIGENVAL and a participation_ratio.json, then renders the
    standard Kohn-Sham level plot where each horizontal bar is coloured by
    the selected localization metric instead of plain black.

    \b
        defectpl pr ksplot --eigenval EIGENVAL --pr-json participation_ratio.json \\
            --vbm 5.20 --cbm 8.10 --espan 1.5 --metric p_ratio
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        raise click.ClickException(
            "matplotlib is required.  Install with: pip install matplotlib"
        )

    from defectpl.io.vasp import read_eigenval_file
    from defectpl.ks_analysis import extract_ksplot_data, plot_ks_with_pr

    try:
        eigenval_data = read_eigenval_file(eigenval, k_idx=kpt_idx)
        ks_data = extract_ksplot_data(eigenval_data, vbm=vbm, cbm=cbm, espan=espan)
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(f"Failed to read EIGENVAL: {exc}")

    with open(pr_json) as fh:
        pr_result = json.load(fh)

    try:
        plot_ks_with_pr(
            ks_data,
            pr_result,
            metric=metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            kpt_idx=kpt_idx,
            title=title,
            output_filename=out,
        )
    except Exception as exc:
        if _verbose:
            raise
        raise click.ClickException(str(exc))

    click.secho(f"Saved: {out}", fg="green", bold=True)


# Register the pr group
main.add_command(pr_group)


# ===========================================================================
# TDM / WAVECAR CLI commands
# ===========================================================================


@click.group("tdm")
@click.pass_context
def tdm_group(ctx):
    """Transition Dipole Moment calculations from VASP WAVECAR files.

    \b
    Sub-commands
    ------------
      calc       Compute TDM between two band indices.
      all        Compute TDMs for all occ→unocc pairs (BZ average).
      cross      Cross-state TDM between two different WAVECARs.
      trim       Write a compact or trimmed WAVECAR for selected bands.
      export     Export WAVECAR metadata to JSON or HDF5.
      plot       Plot TDM results (heatmap / bubble / dashboard / absorption).
    """


@tdm_group.command("calc")
@click.option(
    "--wavecar",
    "wavecar",
    default="WAVECAR",
    show_default=True,
    help="Path to WAVECAR (compressed .gz/.bz2 supported).",
)
@click.option("--ispin", default=1, show_default=True, help="Spin channel (1 or 2).")
@click.option(
    "--iband-i",
    "iband_i",
    required=True,
    type=int,
    help="Initial band index (1-based).",
)
@click.option(
    "--iband-j", "iband_j", required=True, type=int, help="Final band index (1-based)."
)
@click.option("--ibzkpt", default=None, help="IBZKPT file for k-weighting.")
@click.option(
    "--all-kpoints",
    "all_kpoints",
    is_flag=True,
    default=False,
    help="Print TDM at every k-point.",
)
@click.option("--out", default=None, help="Save result JSON to this path.")
@click.pass_context
def tdm_calc(ctx, wavecar, ispin, iband_i, iband_j, ibzkpt, all_kpoints, out):
    """Compute TDM between two specific bands.

    \b
      defectpl tdm calc --iband-i 638 --iband-j 639
      defectpl tdm calc --iband-i 638 --iband-j 639 --ibzkpt IBZKPT --out tdm.json
    """
    from defectpl.physics.tdm import WavecarReader, read_ibzkpt_weights

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(f"Cannot open WAVECAR: {exc}")

    res = wfc.get_tdm_all_kpoints(ispin, iband_i, iband_j)

    if ibzkpt:
        try:
            kw = read_ibzkpt_weights(ibzkpt)
            avg = wfc.get_weighted_avg_tdm(ispin, iband_i, iband_j, kw)
            click.echo(f"\n  BZ-averaged |TDM| = {avg['avg_tdm_magnitude']:.6f} Debye")
            click.echo(
                "  Components (|x|, |y|, |z|) = "
                + "  ".join(f"{v:.4f}" for v in avg["avg_tdm_components"])
                + " Debye"
            )
            click.echo(f"  avg ΔE = {avg['avg_dE']:.4f} eV")
        except Exception as exc:
            click.secho(f"  [warning] Could not k-weight: {exc}", fg="yellow")
    else:
        click.echo(f"\n  Bands: {iband_i} → {iband_j}")
        click.echo(f"  Mean |TDM| = {res['tdm_magnitude'].mean():.6f} Debye")
        click.echo(f"  Max  |TDM| = {res['tdm_magnitude'].max():.6f} Debye")

    if all_kpoints:
        click.echo("\n  k-point breakdown:")
        for i, (mag, dE_val) in enumerate(
            zip(res["tdm_magnitude"], res["dE"]), start=1
        ):
            click.echo(f"    k{i:4d}  |TDM|={mag:.4f} D   ΔE={dE_val:.4f} eV")

    if out:
        import json
        import numpy as np

        def _ser(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out, "w") as fh:
            json.dump({k: _ser(v) for k, v in res.items()}, fh, indent=2)
        click.secho(f"  Result written to: {out}", fg="green")


@tdm_group.command("all")
@click.option("--wavecar", default="WAVECAR", show_default=True)
@click.option("--ispin", default=1, show_default=True)
@click.option(
    "--ibzkpt", default="IBZKPT", show_default=True, help="IBZKPT for k-point weights."
)
@click.option(
    "--mode",
    default="occupation",
    show_default=True,
    type=click.Choice(
        [
            "occupation",
            "energy",
            "band_range",
            "band_list",
            "near_fermi",
            "homo_lumo_range",
            "energy_window",
        ]
    ),
    help="Band selection strategy.",
)
@click.option(
    "--n-occ",
    "n_occ",
    default=10,
    show_default=True,
    help="Number of occupied bands (near_fermi mode).",
)
@click.option(
    "--n-unocc",
    "n_unocc",
    default=10,
    show_default=True,
    help="Number of unoccupied bands (near_fermi mode).",
)
@click.option(
    "--occ-bands",
    "occ_bands",
    default=None,
    help="Occupied band range, e.g. '630-638' (band_range mode).",
)
@click.option(
    "--unocc-bands",
    "unocc_bands",
    default=None,
    help="Unoccupied band range, e.g. '639-650' (band_range mode).",
)
@click.option(
    "--fermi-level",
    "fermi_level",
    default=None,
    type=float,
    help="Fermi level in eV (overrides occupancy from WAVECAR).",
)
@click.option(
    "--min-tdm",
    "min_tdm",
    default=0.0,
    show_default=True,
    type=float,
    help="Discard pairs with |TDM| < this value (Debye).",
)
@click.option(
    "--top",
    "top_n",
    default=10,
    show_default=True,
    type=int,
    help="Number of strongest transitions to print.",
)
@click.option(
    "--out", default="all_tdm.json", show_default=True, help="Output JSON file."
)
@click.pass_context
def tdm_all(
    ctx,
    wavecar,
    ispin,
    ibzkpt,
    mode,
    n_occ,
    n_unocc,
    occ_bands,
    unocc_bands,
    fermi_level,
    min_tdm,
    top_n,
    out,
):
    """Compute TDMs for all occupied→unoccupied band pairs.

    \b
      defectpl tdm all --mode occupation --top 10 --out all_tdm.json
      defectpl tdm all --mode band_range --occ-bands 630-638 --unocc-bands 639-650
    """
    import json
    import numpy as np
    from defectpl.physics.tdm import WavecarReader, read_ibzkpt_weights

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(f"Cannot open WAVECAR: {exc}")

    try:
        kw = read_ibzkpt_weights(ibzkpt)
    except Exception as exc:
        click.secho(
            f"  [warning] Could not read IBZKPT: {exc}. Using equal weights.",
            fg="yellow",
        )
        kw = np.ones(wfc.nkpts)

    occ_range = unocc_range = None
    if occ_bands:
        lo, hi = (int(x) for x in occ_bands.split("-"))
        occ_range = (lo, hi)
    if unocc_bands:
        lo, hi = (int(x) for x in unocc_bands.split("-"))
        unocc_range = (lo, hi)

    click.echo(f"  Computing all transitions (mode={mode})…")
    result = wfc.get_all_transitions(
        ispin=ispin,
        kweights=kw,
        mode=mode if mode != "near_fermi" else "occupation",
        occ_bands=occ_range,
        unocc_bands=unocc_range,
        fermi_level=fermi_level if fermi_level is not None else 0.0,
        min_tdm=min_tdm,
    )

    click.echo(f"\n  Unique band pairs: {result['metadata']['n_unique_pairs']}")
    click.echo(f"\n  Top-{top_n} transitions:")
    for entry in result["strongest_transitions"][:top_n]:
        click.echo(
            f"    {entry['iband_i']:4d} → {entry['iband_j']:4d}  "
            f"|TDM|={entry['avg_tdm_magnitude']:.4f} D  "
            f"ΔE={entry['avg_dE']:.4f} eV"
        )

    def _ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        return obj

    with open(out, "w") as fh:
        json.dump(_ser(result), fh, indent=2)
    click.secho(f"\n  Full result written to: {out}", fg="green")


@tdm_group.command("cross")
@click.option(
    "--wavecar-gs",
    "wavecar_gs",
    default="WAVECAR",
    show_default=True,
    help="Ground-state WAVECAR.",
)
@click.option(
    "--wavecar-es", "wavecar_es", required=True, help="Excited-state WAVECAR."
)
@click.option("--ispin", default=1, show_default=True)
@click.option("--iband-i", "iband_i", required=True, type=int)
@click.option("--iband-j", "iband_j", required=True, type=int)
@click.option("--ibzkpt", default=None, help="IBZKPT for k-weighting.")
@click.option("--out", default=None, help="Output JSON.")
@click.pass_context
def tdm_cross(ctx, wavecar_gs, wavecar_es, ispin, iband_i, iband_j, ibzkpt, out):
    """Compute cross-state TDM (ΔSCF) between two WAVECARs.

    \b
      defectpl tdm cross --wavecar-gs WAVECAR_gs --wavecar-es WAVECAR_es \\
          --iband-i 638 --iband-j 639
    """
    import json
    import numpy as np
    from defectpl.physics.tdm import WavecarReader, read_ibzkpt_weights

    try:
        wfc_gs = WavecarReader(wavecar_gs)
        wfc_es = WavecarReader(wavecar_es)
    except Exception as exc:
        raise click.ClickException(f"Cannot open WAVECAR: {exc}")

    res = wfc_gs.get_tdm_cross_state_all_kpoints(wfc_es, ispin, iband_i, iband_j)

    if ibzkpt:
        kw = read_ibzkpt_weights(ibzkpt)
        w = kw / kw.sum()
        avg = float(np.dot(w, res["tdm_magnitude"]))
        click.echo(f"  BZ-averaged cross-state |TDM| = {avg:.6f} Debye")
    else:
        click.echo(
            f"  Mean cross-state |TDM| = {res['tdm_magnitude'].mean():.6f} Debye"
        )
        click.echo(f"  Max  cross-state |TDM| = {res['tdm_magnitude'].max():.6f} Debye")

    if out:

        def _ser(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out, "w") as fh:
            json.dump({k: _ser(v) for k, v in res.items()}, fh, indent=2)
        click.secho(f"  Result written to: {out}", fg="green")


@tdm_group.command("trim")
@click.option("--wavecar", default="WAVECAR", show_default=True)
@click.option(
    "--bands",
    required=True,
    help="Band indices to keep, comma-separated or range, e.g. '635-639' or '635,636,637'.",
)
@click.option("--out", default="WAVECAR_trim", show_default=True)
@click.option(
    "--compact",
    is_flag=True,
    default=False,
    help="Write compact WAVECAR (only selected band records; auto-detected on read).",
)
@click.pass_context
def tdm_trim(ctx, wavecar, bands, out, compact):
    """Write a trimmed or compact WAVECAR with selected bands only.

    \b
      defectpl tdm trim --bands 635-639 --out WAVECAR_trim
      defectpl tdm trim --bands 630,635,638,639 --compact --out WAVECAR_compact
    """
    from defectpl.physics.tdm import WavecarReader

    if "-" in bands and "," not in bands:
        lo, hi = (int(x) for x in bands.split("-"))
        band_list = list(range(lo, hi + 1))
    else:
        band_list = [int(x) for x in bands.replace("-", ",").split(",") if x.strip()]

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(str(exc))

    kept = wfc.trim_save_wavecar(band_list, outfile=out, compact=compact)
    click.secho(f"  Written {out}  (bands: {sorted(kept)})", fg="green")


@tdm_group.command("export")
@click.option("--wavecar", default="WAVECAR", show_default=True)
@click.option(
    "--bands",
    default=None,
    help="Band indices, e.g. '635-639' or '635,636'. All if omitted.",
)
@click.option(
    "--format",
    "fmt",
    default="json",
    type=click.Choice(["json", "h5"]),
    show_default=True,
)
@click.option(
    "--save-coeffs",
    "save_coeffs",
    is_flag=True,
    default=False,
    help="Include plane-wave coefficients in JSON export.",
)
@click.option("--out", default=None, help="Output file path.")
@click.pass_context
def tdm_export(ctx, wavecar, bands, fmt, save_coeffs, out):
    """Export WAVECAR metadata (and optionally coefficients) to JSON or HDF5.

    \b
      defectpl tdm export --format json --bands 635-640 --out bands.json
      defectpl tdm export --format h5   --bands 635,638,639 --out trim.h5
    """
    from defectpl.physics.tdm import WavecarReader

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(str(exc))

    band_list = None
    if bands is not None:
        if "-" in bands and "," not in bands:
            lo, hi = (int(x) for x in bands.split("-"))
            band_list = list(range(lo, hi + 1))
        else:
            band_list = [int(x) for x in bands.split(",") if x.strip()]

    if fmt == "json":
        out = out or "wavecar_info.json"
        wfc.save_to_json(bands=band_list, outfile=out, save_coeffs=save_coeffs)
    else:
        out = out or "wavecar_trim.h5"
        wfc.save_to_h5(bands=band_list, outfile=out)
    click.secho(f"  Written: {out}", fg="green")


@tdm_group.command("plot")
@click.option(
    "--tdm-json",
    "tdm_json",
    required=True,
    help="JSON file from 'tdm calc' or 'tdm all' command.",
)
@click.option(
    "--plot-type",
    "plot_type",
    default="dashboard",
    type=click.Choice(
        ["dashboard", "heatmap", "bubble", "components", "strip", "absorption"]
    ),
    show_default=True,
)
@click.option(
    "--component",
    default="magnitude",
    type=click.Choice(["magnitude", "x", "y", "z"]),
    help="TDM component for heatmap plot.",
)
@click.option(
    "--sigma",
    default=0.05,
    type=float,
    show_default=True,
    help="Gaussian broadening in eV for absorption plot.",
)
@click.option("--out", default=None, help="Output image file. Display if omitted.")
@click.pass_context
def tdm_plot(ctx, tdm_json, plot_type, component, sigma, out):
    """Plot TDM results from a previously saved JSON file.

    \b
      defectpl tdm plot --tdm-json tdm.json --plot-type dashboard --out dash.pdf
      defectpl tdm plot --tdm-json tdm.json --plot-type absorption --sigma 0.03
    """
    import json
    from defectpl.physics.tdm_viz import (
        plot_tdm_dashboard,
        plot_tdm_heatmap,
        plot_tdm_bubble,
        plot_tdm_components,
        plot_tdm_kpoint_strip,
        plot_tdm_absorption,
    )

    with open(tdm_json) as fh:
        data = json.load(fh)

    import numpy as np

    for key in ("tdm_magnitude", "tdm_components", "E_i", "E_j", "dE", "kvecs"):
        if key in data:
            data[key] = np.array(data[key])

    dispatch = {
        "dashboard": lambda: plot_tdm_dashboard(data, sigma=sigma, outfile=out),
        "heatmap": lambda: plot_tdm_heatmap(data, component=component, outfile=out),
        "bubble": lambda: plot_tdm_bubble(data, outfile=out),
        "components": lambda: plot_tdm_components(data, outfile=out),
        "strip": lambda: plot_tdm_kpoint_strip(data, outfile=out),
        "absorption": lambda: plot_tdm_absorption(data, sigma=sigma, outfile=out),
    }
    dispatch[plot_type]()


main.add_command(tdm_group)


# ---------------------------------------------------------------------------
# IPR commands
# ---------------------------------------------------------------------------


@click.group("ipr")
@click.pass_context
def ipr_group(ctx):
    """Inverse Participation Ratio (IPR) of Kohn-Sham states from WAVECAR.

    \b
    Sub-commands
    ------------
      calc    Compute IPR for selected bands.
      plot    Plot IPR results.
    """


@ipr_group.command("calc")
@click.option("--wavecar", default="WAVECAR", show_default=True)
@click.option("--ispin", default=1, show_default=True)
@click.option("--ibzkpt", default="IBZKPT", show_default=True)
@click.option(
    "--bands",
    default=None,
    help="Band indices, e.g. '635-650' or '635,638,639'. All near-Fermi if omitted.",
)
@click.option(
    "--mode",
    default="near_fermi",
    show_default=True,
    type=click.Choice(
        [
            "all",
            "near_fermi",
            "homo_lumo_range",
            "energy_window",
            "band_range",
            "band_list",
        ]
    ),
)
@click.option("--n-occ", "n_occ", default=10, show_default=True)
@click.option("--n-unocc", "n_unocc", default=10, show_default=True)
@click.option("--fermi-level", "fermi_level", default=None, type=float)
@click.option("--out-json", "out_json", default="ipr_result.json", show_default=True)
@click.option("--out-csv", "out_csv", default="ipr_summary.csv", show_default=True)
@click.pass_context
def ipr_calc(
    ctx,
    wavecar,
    ispin,
    ibzkpt,
    bands,
    mode,
    n_occ,
    n_unocc,
    fermi_level,
    out_json,
    out_csv,
):
    """Compute IPR for all selected Kohn-Sham states.

    \b
      defectpl ipr calc --wavecar WAVECAR --mode near_fermi --n-occ 10 --n-unocc 10
      defectpl ipr calc --bands 635-645 --out-json ipr.json --out-csv ipr.csv
    """
    import numpy as np
    from defectpl.physics.tdm import (
        WavecarReader,
        compute_ipr_all,
        save_ipr_json,
        save_ipr_csv,
        read_ibzkpt_weights,
    )

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(str(exc))

    try:
        kw = read_ibzkpt_weights(ibzkpt)
    except Exception:
        kw = np.ones(wfc.nkpts)

    band_list = None
    if bands is not None:
        if "-" in bands and "," not in bands:
            lo, hi = (int(x) for x in bands.split("-"))
            band_list = list(range(lo, hi + 1))
        else:
            band_list = [int(x) for x in bands.split(",") if x.strip()]

    click.echo("  Computing IPR…")
    result = compute_ipr_all(
        wfc,
        ispin=ispin,
        kweights=kw,
        bands=band_list,
        select_mode=mode,
        n_occ=n_occ,
        n_unocc=n_unocc,
        fermi_level=fermi_level,
    )

    save_ipr_json(result, out_json)
    save_ipr_csv(result, out_csv)

    top = sorted(
        result["band_summary"], key=lambda r: r["weighted_avg_ipr"], reverse=True
    )[:5]
    click.echo("\n  Top-5 most-localised bands (by k-weighted IPR):")
    for r in top:
        click.echo(
            f"    Band {r['iband']:4d}  IPR={r['weighted_avg_ipr']:.4e}  "
            f"E={r['avg_energy']:.3f} eV"
        )
    click.secho(f"\n  JSON: {out_json}   CSV: {out_csv}", fg="green")


@ipr_group.command("plot")
@click.option("--ipr-json", "ipr_json", required=True)
@click.option(
    "--plot-type",
    "plot_type",
    default="scatter",
    type=click.Choice(["scatter", "bar", "heatmap"]),
    show_default=True,
)
@click.option(
    "--fermi-level", "fermi_level", default=0.0, type=float, show_default=True
)
@click.option(
    "--top-n",
    "top_n",
    default=20,
    type=int,
    show_default=True,
    help="Number of bands for bar chart.",
)
@click.option("--out", default=None)
@click.pass_context
def ipr_plot(ctx, ipr_json, plot_type, fermi_level, top_n, out):
    """Plot IPR results from a saved JSON file.

    \b
      defectpl ipr plot --ipr-json ipr_result.json --plot-type scatter
      defectpl ipr plot --ipr-json ipr_result.json --plot-type bar --top-n 15
    """
    import json
    from defectpl.physics.tdm_viz import (
        plot_ipr_scatter,
        plot_ipr_bar,
        plot_ipr_kpoint_heatmap,
    )

    with open(ipr_json) as fh:
        result = json.load(fh)

    if plot_type == "scatter":
        plot_ipr_scatter(result, fermi_level=fermi_level, outfile=out)
    elif plot_type == "bar":
        plot_ipr_bar(result, top_n=top_n, outfile=out)
    else:
        plot_ipr_kpoint_heatmap(result, outfile=out)


main.add_command(ipr_group)


# ---------------------------------------------------------------------------
# ZPL / optical-properties command
# ---------------------------------------------------------------------------


@click.group("zpl")
@click.pass_context
def zpl_group(ctx):
    """Zero-phonon line and radiative lifetime from VASP directories.

    \b
    Sub-commands
    ------------
      calc    Compute ZPL, dQ, Einstein A, and radiative lifetime.
    """


@zpl_group.command("calc")
@click.option("--ground", required=True, help="Ground-state directory.")
@click.option("--excited", required=True, help="Excited-state directory.")
@click.option(
    "--tdm-gg",
    "tdm_gg",
    default=None,
    type=float,
    help="Same-state BZ-averaged |TDM| in Debye.",
)
@click.option(
    "--tdm-ge",
    "tdm_ge",
    default=None,
    type=float,
    help="Cross-state BZ-averaged |TDM| in Debye.",
)
@click.option(
    "--nr", default=2.42, type=float, show_default=True, help="Refractive index."
)
@click.option(
    "--prefer",
    default="oszicar",
    type=click.Choice(["oszicar", "outcar", "vasprun"]),
    show_default=True,
)
@click.option("--out", default=None, help="Save JSON to this path.")
@click.pass_context
def zpl_calc(ctx, ground, excited, tdm_gg, tdm_ge, nr, prefer, out):
    """Compute ZPL, dQ, Einstein A, and radiative lifetime.

    \b
      defectpl zpl calc --ground /data/gs/ --excited /data/es/ --nr 2.65
      defectpl zpl calc --ground /data/gs/ --excited /data/es/ \\
          --tdm-gg 1.5 --tdm-ge 1.3 --nr 2.65 --out optical.json
    """
    import json
    from defectpl.physics.tdm import compute_optical_properties

    try:
        props = compute_optical_properties(
            g_path=ground,
            e_path=excited,
            tdm_gg=tdm_gg,
            tdm_ge=tdm_ge,
            nr=nr,
            prefer_energy=prefer,
        )
    except Exception as exc:
        raise click.ClickException(str(exc))

    click.echo(f"\n  E_ground  = {props['E_ground']:.6f} eV")
    click.echo(f"  E_excited = {props['E_excited']:.6f} eV")
    click.echo(f"  ZPL       = {props['ZPL']:.6f} eV")
    if props["dQ"] is not None:
        click.echo(f"  dQ        = {props['dQ']:.6f} amu^0.5·Å")
    click.echo(f"  n_r       = {nr}")
    for tag, label in [("gg", "Same-state (GG)"), ("ge", "Cross-state (GE)")]:
        tdm = props[f"tdm_{tag}"]
        if tdm is not None:
            A = props[f"A_{tag}"]
            lt = props[f"lifetime_{tag}"]
            click.echo(f"\n  [{label}]")
            click.echo(f"    |TDM|    = {tdm:.6f} Debye")
            click.echo(f"    A        = {A:.4f} MHz")
            click.echo(f"    Lifetime = {lt:.4f} ns")

    if out:
        with open(out, "w") as fh:
            json.dump(
                {k: (float(v) if v is not None else None) for k, v in props.items()},
                fh,
                indent=2,
            )
        click.secho(f"\n  Optical properties saved to: {out}", fg="green")


main.add_command(zpl_group)


# ---------------------------------------------------------------------------
# WFC export command
# ---------------------------------------------------------------------------


@click.group("wfc")
@click.pass_context
def wfc_group(ctx):
    """Real-space wavefunction export from WAVECAR.

    \b
    Sub-commands
    ------------
      save    Export to CHGCAR (.vasp) and/or VESTA project (.vesta).
    """


@wfc_group.command("save")
@click.option("--wavecar", default="WAVECAR", show_default=True)
@click.option(
    "--poscar",
    default=None,
    help="POSCAR/CONTCAR for structure. Auto-detected if omitted.",
)
@click.option("--ispin", default=1, show_default=True)
@click.option("--ikpt", default=1, show_default=True)
@click.option("--iband", required=True, type=int)
@click.option(
    "--quantity",
    default="density",
    type=click.Choice(["density", "real", "imag"]),
    show_default=True,
)
@click.option(
    "--vesta",
    "write_vesta",
    is_flag=True,
    default=False,
    help="Also write a .vesta project file.",
)
@click.option(
    "--out", default=None, help="Output basename (default: wfc_band{iband}.vasp)."
)
@click.pass_context
def wfc_save(ctx, wavecar, poscar, ispin, ikpt, iband, quantity, write_vesta, out):
    """Export a real-space KS state to CHGCAR + optional VESTA project.

    \b
      defectpl wfc save --iband 638 --vesta
      defectpl wfc save --iband 639 --quantity real --out psi_re.vasp
    """
    from defectpl.physics.tdm import WavecarReader
    from defectpl.physics.tdm_viz import save_wfc_vasp, save_wfc_vesta
    from defectpl.io.wavecar import get_structure, read_poscar
    from pathlib import Path

    try:
        wfc = WavecarReader(wavecar)
    except Exception as exc:
        raise click.ClickException(str(exc))

    try:
        if poscar:
            structure = read_poscar(poscar)
        else:
            structure = get_structure(".", relaxed=True)
    except Exception as exc:
        raise click.ClickException(f"Cannot read structure: {exc}")

    out_vasp = Path(out) if out else Path(f"wfc_band{iband}.vasp")
    save_wfc_vasp(
        wfc, ispin, ikpt, iband, structure, outfile=out_vasp, quantity=quantity
    )
    if write_vesta:
        out_vesta = out_vasp.with_suffix(".vesta")
        save_wfc_vesta(
            wfc, ispin, ikpt, iband, structure, vasp_file=out_vasp, outfile=out_vesta
        )


main.add_command(wfc_group)


if __name__ == "__main__":
    main()
