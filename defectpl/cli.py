# -*- coding: utf-8 -*-
"""
Command Line Interface (CLI) application layer managing execution workflows
for the defectpl suite, covering photoluminescence lineshapes, structural
displacements, Configuration Coordinate Diagrams (CCD), and phonon properties.
"""

import json
from pathlib import Path
import click


@click.group()
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
    help="Broadening damping parameter managing electronic state correlation lifespans.",
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
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run PL calculations using atomic structural shifts (Displacement Mode)."""
    from defectpl.phonon import read_band_yaml
    from pymatgen.core import Structure
    from defectpl.vasp_wrapper import calc_dR
    from defectpl.defectpl import Photoluminescence
    from monty.serialization import dumpfn

    try:
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
            max_energy=5.0,
            sigma=6e-3,
        )
        click.echo("Photoluminescence engine data properties calculated successfully.")

        if json_out:
            dumpfn(pl_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoluminescence class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            pl_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )

    except Exception as exc:
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
    help="Broadening damping parameter managing electronic state correlation lifespans.",
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
    json_out,
    plot_all,
    fig_format,
    iylim,
    max_freq,
):
    """Run PL calculations using force-difference vectors at vertical excitation (Force Mode)."""
    from defectpl.phonon import read_band_yaml
    from defectpl.vasp_wrapper import prepare_dF_files
    from defectpl.defectpl import Photoluminescence
    from monty.serialization import dumpfn

    try:
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
            max_energy=5.0,
            sigma=6e-3,
        )
        click.echo("Photoluminescence engine data properties calculated successfully.")

        if json_out:
            dumpfn(pl_engine, json_out, indent=2)
            click.echo(
                f"Serialized Photoluminescence class properties safely to {json_out}"
            )

        if plot_all:
            parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None
            pl_engine.generate_plots(
                out_dir=out_dir,
                fig_format=fig_format,
                iylim=parsed_iylim,
                max_freq=max_freq,
            )

    except Exception as exc:
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


# =====================================================================
# INDIVIDUAL PLOT CONTROLLER COMMAND
# =====================================================================


@main.command(name="plot")
@click.argument("json_file", type=click.Path(exists=True))
@click.option(
    "--type",
    "-t",
    "plot_type",
    type=click.Choice(
        ["intensity", "mode", "partial_energy", "all"], case_sensitive=False
    ),
    required=True,
    help="The specific individual graphic component layout to plot from the data file.",
)
@click.option(
    "--out_dir",
    default="./",
    help="Output destination path mapping location to dump the generated graphic.",
)
@click.option(
    "--fmt",
    default="pdf",
    help="Export graphic file standard extension layout (e.g., pdf, png, svg).",
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
def plot_individual(json_file, plot_type, out_dir, fmt, iylim, max_freq):
    """Deserialize a saved property JSON file and render standalone visualization metrics plots."""
    from defectpl.defectpl import Photoluminescence
    from defectpl.plot import Plotter
    from monty.serialization import loadfn

    try:
        click.echo(f"Loading data profile records from: {json_file}")
        # Deserializes complex data patterns seamlessly via MSONable schema hooks
        pl_engine = loadfn(json_file)
        if not isinstance(pl_engine, Photoluminescence):
            raise ValueError(
                "The deserialized records did not resolve into a standard Photoluminescence engine."
            )

        plotter = Plotter(pl_engine)
        parsed_iylim = [float(x) for x in iylim.split(",")] if iylim else None

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if plot_type.lower() in ["intensity", "all"]:
            plotter.plot_intensity_vs_penergy(
                out_dir=out_dir, fig_format=fmt, iylim=parsed_iylim
            )
            click.echo(f"Rendered Intensity vs Photon Energy plot in {out_dir}")

        if plot_type.lower() in ["mode", "all"]:
            plotter.plot_penergy_vs_pmode(
                out_dir=out_dir, fig_format=fmt, max_freq=max_freq
            )
            click.echo(
                f"Rendered Partial Energy / S_k distribution vs Mode plot in {out_dir}"
            )

        if plot_type.lower() in ["partial_energy", "all"]:
            # Check if specialized plot engine components exist in local module
            if hasattr(plotter, "plot_partial_energy_vs_penergy"):
                plotter.plot_partial_energy_vs_penergy(out_dir=out_dir, fig_format=fmt)
                click.echo(f"Rendered Partial Energy vs Energy plot in {out_dir}")
            elif plot_type.lower() == "partial_energy":
                click.echo(
                    "Warning: 'plot_partial_energy_vs_penergy' method not found in Plotter engine.",
                    err=True,
                )

    except Exception as exc:
        raise click.ClickException(
            f"Failed to generate custom standalone graphic: {exc}"
        )


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
    from defectpl.vasp_wrapper import calc_delta_Q

    try:
        s1 = Structure.from_file(structure1)
        s2 = Structure.from_file(structure2)
        delta_q = calc_delta_Q(s1, s2)
    except Exception as exc:
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
    from defectpl.vasp_wrapper import run_dynamic_yaml_comparison

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
    from defectpl.vasp_wrapper import run_kohn_sham_analysis

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
        raise click.ClickException(
            f"Kohn-Sham calculation plotting tracking layer crashed: {exc}"
        )


# Link the nested PL group structure into our root application workspace
main.add_command(pl_group)


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

    from defectpl.vasp import read_eigenval_file
    from defectpl.ks_analysis import extract_ksplot_data, plot_ks_with_pr

    try:
        eigenval_data = read_eigenval_file(eigenval, k_idx=kpt_idx)
        ks_data = extract_ksplot_data(eigenval_data, vbm=vbm, cbm=cbm, espan=espan)
    except Exception as exc:
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
        raise click.ClickException(str(exc))

    click.secho(f"Saved: {out}", fg="green", bold=True)


# Register the pr group
main.add_command(pr_group)


if __name__ == "__main__":
    main()
