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
# PHOTOLUMINESCENCE ENGINE COMMANDS
# =====================================================================

@main.command(name="run")
@click.option(
    "--band_yaml",
    default="./band.yaml",
    help="Phonopy band.yaml configuration file track destination path.",
)
@click.option(
    "--contcar_gs",
    default="./CONTCAR_gs",
    help="Pymatgen readable Ground State equilibrium configuration structure file path.",
)
@click.option(
    "--contcar_es",
    default="./CONTCAR_es",
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
    help="Energy value designating Zero-Phonon Line baseline transitions boundary in eV.",
)
@click.option(
    "--gamma",
    default=2.0,
    type=float,
    help="Broadening damping parameter managing electronic state correlation lifespans.",
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
def run(
    band_yaml,
    contcar_gs,
    contcar_es,
    out_dir,
    ezpl,
    gamma,
    plot_all,
    fig_format,
):
    """Execute high-level defect photoluminescence multi-mode spectral profile engines."""
    from pymatgen.core import Structure
    from defectpl.phonon import read_band_yaml
    from defectpl.utils import calc_dR
    from defectpl.defectpl import Photoluminescence

    try:
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml)
        struct_gs = Structure.from_file(contcar_gs)
        struct_es = Structure.from_file(contcar_es)
        dR = calc_dR(struct_gs, struct_es)

        pl_engine = Photoluminescence(
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=dR,
            EZPL=ezpl,
            gamma=gamma,
            max_energy=5.0,
            sigma=6e-3
        )
        click.echo("Photoluminescence engine data properties calculated successfully.")
        
        if plot_all:
            pl_engine.generate_plots(out_dir=out_dir, fig_format=fig_format)
            
    except Exception as exc:
        raise click.ClickException(f"Calculation pipeline failure encountered: {exc}")


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
    from defectpl.utils import calc_delta_Q

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
@click.option("--ezpl", required=True, type=float, help="Zero-phonon line energy value (eV).")
@click.option("--w1", required=True, type=float, help="Vibrational frequency tracking excited state wells (meV).")
@click.option("--w2", required=True, type=float, help="Vibrational frequency tracking ground state wells (meV).")
@click.option("--dq_val", required=True, type=float, help="Generalized coordinate offset delta Q.")
@click.option("--temp", default=300.0, type=float, help="System operational temperature parameter in Kelvin.")
@click.option("--e0", default=0.0, type=float, help="Energy evaluation grid starting floor parameter (eV).")
@click.option("--de", default=0.001, type=float, help="Grid channel sampling increment step dimension parameter (eV).")
@click.option("--points", default=5000, type=int, help="Total integrated point matrix rows arrays dimension count.")
@click.option("--nn1", default=22, type=int, help="Maximum quantum state boundary index cut for excited wells.")
@click.option("--nn2", default=52, type=int, help="Maximum quantum state boundary index cut for ground wells.")
@click.option("--plot", is_flag=True, default=False, help="Launches visualizer mapping normalized shapes.")
@click.option("--save_prefix", default="vibrational_1d", help="Prefix for files output during metrics serialization.")
def spectra1d(ezpl, w1, w2, dq_val, temp, e0, de, points, nn1, nn2, plot, save_prefix):
    """Run decoupled analytical 1D Franck-Condon harmonic approximation lineshape tracks."""
    from defectpl.defectpl import VibrationalSpectra1D

    try:
        spec = VibrationalSpectra1D(
            EZPL=ezpl, w1_meV=w1, w2_meV=w2, DQ=dq_val, T=temp, E0=e0, dE=de, M=points, NN1=nn1, NN2=nn2
        )
        click.echo("Computing transition metrics under overlap matrices rules...")
        spec.compute_lineshape()
        spec.get_peak_position()
        spec.get_fwhm()
        
        spec.save_results(
            overlap_file=f"{save_prefix}_overlap.json",
            lineshape_file=f"{save_prefix}_lineshape.json"
        )
        if plot:
            spec.plot_lineshape(save_file=f"{save_prefix}_plot.pdf")
    except Exception as exc:
        raise click.ClickException(f"Analytical processing execution failed: {exc}")


# =====================================================================
# CONFIGURATION COORDINATE DIAGRAM (CCD) PIPELINES
# =====================================================================

@main.command(name="setup-ccd")
@click.option("--gs", required=True, type=click.Path(exists=True), help="Reference ground structure configuration.")
@click.option("--es", required=True, type=click.Path(exists=True), help="Reference excited structure configuration.")
@click.option("--out_dir", default="./ccd_calculations", help="Root directory endpoint mapping out workspace calculations.")
@click.option("--tmpl_gs", required=True, type=click.Path(exists=True), help="Directory hosting ground VASP parameter scripts.")
@click.option("--tmpl_es", required=True, type=click.Path(exists=True), help="Directory hosting excited VASP parameter scripts.")
@click.option("--steps", default="-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2", help="Comma-separated fraction list tracing interpolation paths.")
def setup_ccd(gs, es, out_dir, tmpl_gs, tmpl_es, steps):
    """Generate linear interpolation structure configuration spaces for automated VASP execution parameters."""
    from pymatgen.core import Structure
    from defectpl.vasp_wrapper import generate_ccd_calculations

    try:
        s_gs = Structure.from_file(gs)
        s_es = Structure.from_file(es)
        displacements = [float(x.strip()) for x in steps.split(",")]

        generate_ccd_calculations(
            gs_structure=s_gs,
            es_structure=s_es,
            displacements=displacements,
            output_dir=out_dir,
            ground_template_dir=tmpl_gs,
            excited_template_dir=tmpl_es,
        )
        click.echo(f"Interpolated task configuration structures tree setup complete at: {out_dir}")
    except Exception as exc:
        raise click.ClickException(f"Calculations generation workflow initialization failed: {exc}")


@main.command(name="analyze-ccd")
@click.option("--gs", required=True, type=click.Path(exists=True), help="Baseline structural geometry configuration (GS).")
@click.option("--es", required=True, type=click.Path(exists=True), help="Baseline structural geometry configuration (ES).")
@click.option("--gs_runs", required=True, help="Space-separated paths targeting completed ground state xml run parameters.")
@click.option("--es_runs", required=True, help="Space-separated paths targeting completed excited state xml run parameters.")
@click.option("--de", default=0.0, type=float, help="Energy gap minimum offset scalar separation factor (eV).")
@click.option("--save_plot", default=None, help="If provided, exports the diagram visualization path layout safely.")
def analyze_ccd(gs, es, gs_runs, es_runs, de, save_plot):
    """Fit calculated Potential Energy Surfaces data arrays, extract well parameters, and report metrics."""
    from pymatgen.core import Structure
    from defectpl.vasp_wrapper import analyze_ccd_framework

    try:
        s_gs = Structure.from_file(gs)
        s_es = Structure.from_file(es)
        
        paths_gs = [p.strip() for p in gs_runs.split(" ") if p.strip()]
        paths_es = [p.strip() for p in es_runs.split(" ") if p.strip()]

        w_g, w_e = analyze_ccd_framework(
            gs_structure=s_gs,
            es_structure=s_es,
            ground_vaspruns=paths_gs,
            excited_vaspruns=paths_es,
            dE=de,
            save_plot=save_plot,
        )
        click.echo(f"\nFitted Harmonic Well Parameters Found:")
        click.echo(f"Ground state effective phonon frequency energy: {w_g:.4f} eV")
        click.echo(f"Excited state effective phonon frequency energy: {w_e:.4f} eV")
    except Exception as exc:
        raise click.ClickException(f"Failed to fit potential energy curvature points: {exc}")


# =====================================================================
# DATA BENCHMARKING AND UTILITY METRICS COMMANDS
# =====================================================================

@main.command(name="compare-json")
@click.option("--files", required=True, help="Space-separated file lists targeting processed property JSON files.")
@click.option("--xmin", default=None, type=float, help="Window boundary minimum cutoff limit for photon energy range.")
@click.option("--xmax", default=None, type=float, help="Window boundary maximum cutoff limit for photon energy range.")
@click.option("--legends", default=None, help="Comma-separated labels mapping structural targets sequentially.")
@click.option("--out_dir", default="./", help="Directory destination track location to drop compared figures.")
@click.option("--fmt", default="pdf", help="Image output compression extension style layout.")
def compare_json(files, xmin, xmax, legends, out_dir, fmt):
    """Generate comparative visualization graphs detailing differences across serialized static property JSON files."""
    from defectpl.plot import comparepl

    try:
        target_files = [Path(f.strip()) for f in files.split(" ") if f.strip()]
        labels = [l.strip() for l in legends.split(",")] if legends else None
        xlim = (xmin, xmax) if (xmin is not None and xmax is not None) else None

        comparepl(
            properties_files=target_files,
            xlim=xlim,
            legends=labels,
            out_dir=out_dir,
            fig_format=fmt,
        )
        click.echo(f"Comparative property spectrum graph compiled successfully.")
    except Exception as exc:
        raise click.ClickException(f"Comparison array aggregation processing failure encountered: {exc}")


@main.command(name="compare-yaml")
@click.option("--yamls", required=True, help="Space-separated file paths listing phonopy band.yaml configuration inputs.")
@click.option("--gs", required=True, type=click.Path(exists=True), help="Ground state structure template (CONTCAR).")
@click.option("--es", required=True, type=click.Path(exists=True), help="Excited state structure template (CONTCAR).")
@click.option("--out_dir", default="./", help="Workspace path to save comparisons charts profile graphs.")
@click.option("--ezpl", default=1.95, type=float, help="Zero-phonon line transition point (eV).")
@click.option("--gamma", default=2.0, type=float, help="Broadening parameter factor.")
@click.option("--xmin", default=1.0, type=float, help="Minimum energy domain layout bound limit.")
@click.option("--xmax", default=3.0, type=float, help="Maximum energy domain layout bound limit.")
@click.option("--file_name", default="compare_yaml_intensity.pdf", help="Output plot filename template destination.")
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
            file_name=file_name
        )
        click.echo(f"Dynamic execution spectra comparison chart saved successfully to {out_path}.")
    except Exception as exc:
        raise click.ClickException(f"Dynamic multi-yaml calculation task execution dropped: {exc}")


# =====================================================================
# PHONON ANALYSIS COMMANDS (eV Metrics Scale)
# =====================================================================

@main.command(name="phonon-fc")
@click.argument("vasrun_xml", type=click.Path(exists=True))
@click.option("--hdf5", is_flag=True, default=False, help="Write output force constants using HDF5 format structures.")
@click.option("--log_level", default=1, type=int, help="Verbosity log tracking scale values.")
def phonon_fc(vasrun_xml, hdf5, log_level):
    """Parse a vasprun.xml file and write extracted force constants to a FORCE_CONSTANTS file."""
    from defectpl.phonon import create_force_constants_from_vasprun
    
    status = create_force_constants_from_vasprun(vasrun_xml, is_hdf5=hdf5, log_level=log_level)
    if status == 0:
        click.echo("Force constants written successfully.")
    else:
        raise click.ClickException("Phonopy internal processor failed to write force constants matrix.")


@main.command(name="phonon-symm")
@click.option("--poscar", default="./POSCAR", help="Target VASP structural base unit cell geometry filepath.")
@click.option("--fc", default=None, help="Optional destination path tracker targeting precalculated FORCE_CONSTANTS.")
@click.option("--fs", default=None, help="Optional destination path tracker targeting precalculated FORCE_SETS files.")
@click.option("--dim", default="1 1 1", help="Space-separated scaling factors defining supercell dimensions.")
@click.option("--symprec", default=1e-5, type=float, help="Symmetry recognition grid matching tolerance constraints.")
def phonon_symm(poscar, fc, fs, dim, symprec):
    """Evaluate point-group irreducible representations (irreps) metrics at the Gamma point."""
    from defectpl.phonon import calculate_phonon_symmetries

    if not fc and not fs:
        raise click.UsageError("Execution halted: Provide at least one valid data file source tracking --fc or --fs data entries.")

    try:
        symm_results = calculate_phonon_symmetries(
            unitcell_path=poscar,
            force_constants_path=fc,
            force_sets_path=fs,
            dimension=dim,
            symprec=symprec
        )
        
        click.echo("\nAll the IRs are saved into irreps.yaml file.")
        
    except Exception as exc:
        raise click.ClickException(f"Symmetry parser pipeline execution failed: {exc}")

@main.command(name="phonon-band")
@click.option("--poscar", default="./POSCAR", help="Target VASP structural base unit cell geometry filepath.")
@click.option("--fc", default="./FORCE_CONSTANTS", help="Source force constants tracking parameter file matrix.")
@click.option("--dim", default="1 1 1", help="Space-separated scaling factors defining supercell dimensions.")
@click.option("--out", default="band.yaml", help="Destination file path where the compiled Phonopy YAML structure will be dropped.")
def phonon_band(poscar, fc, dim, out):
    """Evaluate phonon properties at the Gamma point from a FORCE_CONSTANTS file and write to a band.yaml file."""
    from defectpl.phonon import calculate_gamma_phonon_to_band_yaml
    
    try:
        calculate_gamma_phonon_to_band_yaml(
            unitcell_filename=poscar,
            force_constants_filename=fc,
            dimension=dim,
            output_filename=out
        )
        click.echo(f"Phonon path calculations complete. Records dumped safely to: {out}")
    except Exception as exc:
        raise click.ClickException(f"Failed to compile single-point band structure output: {exc}")


@main.command(name="phonon-parse")
@click.argument("band_yaml", type=click.Path(exists=True))
@click.option("--json_out", default="gamma_phonon_properties.json", help="Destination output filename tracker to drop JSON dump data records.")
def phonon_parse(band_yaml, json_out):
    """Parse a band.yaml file to convert all frequencies to eV and serialize into an MSONable JSON data model."""
    from defectpl.phonon import extract_gamma_phonon_data
    
    try:
        phonon_model = extract_gamma_phonon_data(band_yaml)
        p = Path(json_out)
        p.write_text(phonon_model.to_json(), encoding="utf-8")
        
        click.echo(f"Successfully processed {phonon_model.nmodes} modes tracking {phonon_model.natoms} atoms.")
        click.echo(f"Energetic values converted to eV and database profile written to: {json_out}")
    except Exception as exc:
        raise click.ClickException(f"Parsing engine dropped properties evaluation tasks: {exc}")


# =====================================================================
# KOHN-SHAM LEVEL VISUALIZATION COMMANDS
# =====================================================================

@main.command(name="ksplot")
@click.argument("eigenval", type=click.Path(exists=True))
@click.option("--vbm", required=True, type=float, help="Energy value of the Valence Band Maximum (eV).")
@click.option("--cbm", required=True, type=float, help="Energy value of the Conduction Band Minimum (eV).")
@click.option("--espan", default=1.0, type=float, help="Energy canvas buffer padding depth beyond band edges (eV).")
@click.option("--kidx", default=0, type=int, help="Target sequential k-point list array index index grid target.")
@click.option("--out_img", default="ks_plot.png", help="File layout path target endpoint to drop final generated graphic.")
@click.option("--out_json", default=None, help="Optional destination to export serialized MSONable JSON data model records.")
def ksplot(eigenval, vbm, cbm, espan, kidx, out_img, out_json):
    """
    Extract, resolve degeneracies, and plot spin-polarized Kohn-Sham electronic states near the bandgap.

    Usage:\n
      defectpl ksplot ./EIGENVAL --vbm 9.6747 --cbm 13.7934 --kidx 0 --out_img gap_levels.png
    """
    from defectpl.vasp_wrapper import run_kohn_sham_analysis

    try:
        click.echo(f"Initializing parsing matrix structures tracking target point: {eigenval}")
        
        run_kohn_sham_analysis(
            eigenval_path=eigenval,
            vbm=vbm,
            cbm=cbm,
            espan=espan,
            k_idx=kidx,
            output_img=out_img,
            output_json=out_json
        )
        
        click.echo(f"Successfully generated Kohn-Sham electronic levels matrix diagram at: {out_img}")
        if out_json:
            click.echo(f"Exported MSONable state database models JSON records to: {out_json}")
            
    except Exception as exc:
        raise click.ClickException(f"Kohn-Sham calculation plotting tracking layer crashed: {exc}")


if __name__ == "__main__":
    main()