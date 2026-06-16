# Script to test PL functionality of DefectPL

import gzip
import os
import shutil
from pathlib import Path
from defectpl.defectpl import Photoluminescence
from defectpl.phonon import read_band_yaml
from defectpl.utils import extract_important_properties
from defectpl.io.vasp import calc_dR, prepare_dF_files
from monty.serialization import dumpfn
from pymatgen.core import Structure


def clean_existing_output_directories(base_out_path: Path):
    """Safely wipes and resets the root evaluation directories before execution

    to prevent artifacts or old diagnostic runs from polluting new calculations.
    """
    if base_out_path.exists():
        print(f"==> [CLEANUP] Found existing target directory: {base_out_path}")
        print("              Wiping contents for a fresh execution baseline...")
        shutil.rmtree(base_out_path)

    # Recreate clean base path
    base_out_path.mkdir(parents=True, exist_ok=True)


def prepare_and_unzip_data(
    source_gz_path: Path, base_out_path: Path, suffix: str = ""
) -> Path:
    """Creates a 'data' directory inside base_out_path, copies the source .gz file

    into it, unzips it with an optional suffix to avoid naming conflicts,
    and returns the Path to the unzipped file.
    """
    target_data_dir = base_out_path / "data"
    target_data_dir.mkdir(parents=True, exist_ok=True)

    # Clean the suffix of any slashes (e.g., 'dfpt/hse06' -> 'dfpt_hse06')
    # to prevent unintended nested directory creation or FileNotFoundError
    safe_suffix = suffix.replace("/", "_") if suffix else ""

    copied_gz_name = (
        f"{source_gz_path.stem}_{safe_suffix}.gz"
        if safe_suffix
        else source_gz_path.name
    )
    unzipped_name = (
        f"{source_gz_path.stem}_{safe_suffix}" if safe_suffix else source_gz_path.stem
    )

    copied_gz_path = target_data_dir / copied_gz_name
    unzipped_path = target_data_dir / unzipped_name

    shutil.copy(source_gz_path, copied_gz_path)

    with gzip.open(copied_gz_path, "rb") as f_in:
        with open(unzipped_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    copied_gz_path.unlink()
    return unzipped_path


def execute_pipeline(
    pipeline_name: str,
    out_path: Path,
    frequencies,
    eigenvectors,
    masses,
    dR=None,
    dF=None,
    ezpl: float = 1.945,
    gamma: float = 2.0,
    fig_format: str = "svg",
):
    """Executes a single Photoluminescence pipeline, exports diagnostic plots,

    summaries, and gz-compressed JSON configurations.
    """
    print(f"\n--- Running PL calculations for {pipeline_name} pipeline ---")
    outdir = out_path / pipeline_name
    os.makedirs(outdir, exist_ok=True)

    print("Running PL engine...")
    pl_engine = Photoluminescence(
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        masses=masses,
        dR=dR,
        dF=dF,
        EZPL=ezpl,
        gamma=gamma,
        max_energy=5.0,
        sigma=6e-3,
    )

    print(f"Generating and exporting diagnostic plots as .{fig_format} files...")
    pl_engine.generate_plots(out_dir=outdir, fig_format=fig_format)

    print("Extracting important properties for summary output...")
    # Converted Path to string for strict compatibility with older package backends
    extract_important_properties(
        pl_engine, filename=str(outdir / "important_properties.txt")
    )

    # 1. Write the raw JSON structure
    output_json_path = outdir / "properties.json"
    dumpfn(pl_engine, str(output_json_path), indent=4)

    # 2. Compress the JSON file into a .gz archive
    output_gz_path = outdir / "properties.json.gz"
    print(f"Compressing state record tracking matrix to: {output_gz_path.name}")
    with open(output_json_path, "rb") as f_in:
        with gzip.open(output_gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # 3. Remove the uncompressed file to save space
    output_json_path.unlink()
    print(
        f"Data state records successfully exported and compressed to: {output_gz_path}"
    )


def run_pl_analysis(
    data_path: Path,
    out_path: Path,
    system_dir: str = "NV_diamond",
    dfpt_dir: str = "dfpt",
    gs_dir: str = "gs",
    abs_dir: str = "abs",
    zpl_dir: str = "zpl",
    ems_dir: str = "ems",
    ezpl: float = 1.945,
    gamma: float = 2.0,
    fig_format: str = "svg",
):
    """Orchestrates data unzipping and pipeline executions with configurable

    internal system subdirectories. Handles automatic cleanup.
    """
    extracted_files = []
    system_base_path = data_path / system_dir

    try:
        print(
            f"\n========================================================================\n"
            f"Preparing and extracting data files for system configuration: {system_dir}\n"
            f"========================================================================"
        )

        # 1. Unzip the core files needed across all pipelines
        band_yaml_path = prepare_and_unzip_data(
            system_base_path / dfpt_dir / "band.yaml.gz",
            out_path,
            suffix=dfpt_dir,
        )
        extracted_files.append(band_yaml_path)

        outcar_gs_path = prepare_and_unzip_data(
            system_base_path / gs_dir / "OUTCAR.gz", out_path, suffix=gs_dir
        )
        extracted_files.append(outcar_gs_path)

        outcar_abs_path = prepare_and_unzip_data(
            system_base_path / abs_dir / "OUTCAR.gz", out_path, suffix=abs_dir
        )
        extracted_files.append(outcar_abs_path)

        contcar_gs_path = prepare_and_unzip_data(
            system_base_path / gs_dir / "CONTCAR.gz", out_path, suffix=gs_dir
        )
        extracted_files.append(contcar_gs_path)

        contcar_zpl_path = prepare_and_unzip_data(
            system_base_path / zpl_dir / "CONTCAR.gz", out_path, suffix=zpl_dir
        )
        extracted_files.append(contcar_zpl_path)

        # 2. Extract specific OUTCAR files for the zpl_ems_force_mode pipeline
        outcar_ems_path = prepare_and_unzip_data(
            system_base_path / ems_dir / "OUTCAR.gz", out_path, suffix=ems_dir
        )
        extracted_files.append(outcar_ems_path)

        outcar_zpl_path = prepare_and_unzip_data(
            system_base_path / zpl_dir / "OUTCAR.gz", out_path, suffix=zpl_dir
        )
        extracted_files.append(outcar_zpl_path)

        # Parse global configuration parameters
        print("Parsing phonon configuration parameters...")
        frequencies, eigenvectors, masses = read_band_yaml(band_yaml_path)

        # ==========================================
        # Pipeline 1: abs_gs_force_mode
        # ==========================================
        print(f"Extracting vertical force differences ({gs_dir} vs {abs_dir})....")
        dF_abs_gs = prepare_dF_files(str(outcar_gs_path), str(outcar_abs_path))
        execute_pipeline(
            pipeline_name="abs_gs_force_mode",
            out_path=out_path,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dF=dF_abs_gs,
            ezpl=ezpl,
            gamma=gamma,
            fig_format=fig_format,
        )

        # ==========================================
        # Pipeline 2: gs_zpl_disp_mode
        # ==========================================
        print(f"Parsing atomic structural profiles ({gs_dir} vs {zpl_dir})...")
        struct_gs = Structure.from_file(str(contcar_gs_path))
        struct_zpl = Structure.from_file(str(contcar_zpl_path))
        dR = calc_dR(struct_gs, struct_zpl)

        execute_pipeline(
            pipeline_name="gs_zpl_disp_mode",
            out_path=out_path,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dR=dR,
            ezpl=ezpl,
            gamma=gamma,
            fig_format=fig_format,
        )

        # ==========================================
        # Pipeline 3: zpl_ems_force_mode
        # ==========================================
        print(f"Extracting vertical force differences ({ems_dir} vs {zpl_dir})...")
        dF_zpl_ems = prepare_dF_files(str(outcar_ems_path), str(outcar_zpl_path))
        execute_pipeline(
            pipeline_name="zpl_ems_force_mode",
            out_path=out_path,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            masses=masses,
            dF=dF_zpl_ems,
            ezpl=ezpl,
            gamma=gamma,
            fig_format=fig_format,
        )

    finally:
        print("\n--- Initiating post-calculation cleanup ---")
        for file_path in extracted_files:
            if file_path.exists():
                print(f"Removing temporary file: {file_path.name}")
                file_path.unlink()
        print("Cleanup completed successfully.")


# --- Execution Entry Point ---
if __name__ == "__main__":

    DATA_DIR = Path(
        "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/data"
    )

    # --------------------------------------------------------------------------
    # PBE Functional Examples
    # --------------------------------------------------------------------------

    # Example 1: Calculation with PBE functional and complete electron excitation
    EX1_OUT = Path(
        "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/examples/NV_diamond/pbe_out"
    )
    clean_existing_output_directories(EX1_OUT)
    run_pl_analysis(
        data_path=DATA_DIR,
        out_path=EX1_OUT,
        system_dir="NV_diamond",
        dfpt_dir="dfpt",
        gs_dir="gs",
        abs_dir="abs",
        zpl_dir="zpl",
        ems_dir="ems",
        ezpl=1.750,
    )

    # Example 2: Calculation with PBE functional and fractional electron excitation
    EX2_OUT = Path(
        "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/examples/NV_diamond/frac_pbe_out"
    )
    clean_existing_output_directories(EX2_OUT)
    run_pl_analysis(
        data_path=DATA_DIR,
        out_path=EX2_OUT,
        system_dir="NV_diamond",
        dfpt_dir="dfpt",
        gs_dir="gs",
        abs_dir="frac_abs",
        zpl_dir="frac_zpl",
        ems_dir="frac_ems",
        ezpl=1.729,
    )

    # --------------------------------------------------------------------------
    # HSE06 Functional Examples
    # --------------------------------------------------------------------------

    # Example 3: Calculation with HSE06 functional and complete electron excitation
    EX3_OUT = Path(
        "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/examples/NV_diamond/hse06_out"
    )
    clean_existing_output_directories(EX3_OUT)
    run_pl_analysis(
        data_path=DATA_DIR,
        out_path=EX3_OUT,
        system_dir="NV_diamond",
        dfpt_dir="dfpt/hse06",
        gs_dir="gs/hse06",
        abs_dir="abs/hse06",
        zpl_dir="zpl/hse06",
        ems_dir="ems/hse06",
        ezpl=1.991,
    )

    # Example 4: Calculation with HSE06 functional and fractional electron excitation
    EX4_OUT = Path(
        "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/examples/NV_diamond/frac_hse06_out"
    )
    clean_existing_output_directories(EX4_OUT)
    run_pl_analysis(
        data_path=DATA_DIR,
        out_path=EX4_OUT,
        system_dir="NV_diamond",
        dfpt_dir="dfpt/hse06",
        gs_dir="gs/hse06",
        abs_dir="frac_abs/hse06",
        zpl_dir="frac_zpl/hse06",
        ems_dir="frac_ems/hse06",
        ezpl=2.190,
    )
