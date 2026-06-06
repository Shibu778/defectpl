# Script to aggregate PL important properties into a summary table

import os
import re
from pathlib import Path
import pandas as pd

# Define paths
BASE_OUT_DIR = Path(
    "/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/tests_out/NV_diamond"
)

# The 4 case folders we want to aggregate
TARGET_CASES = ["pbe_out", "frac_pbe_out", "hse06_out", "frac_hse06_out"]

# The 3 physics pipelines run inside each case folder
PIPELINES = ["abs_gs_force_mode", "gs_zpl_disp_mode", "zpl_ems_force_mode"]


def parse_properties_file(file_path: Path) -> dict:
    """Parses key-value pairs from important_properties.txt."""
    data = {}
    if not file_path.exists():
        return data

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Split on the first colon
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            # Try to convert numeric values to float where possible
            try:
                # Remove common units if present (e.g., 'eV', 'A^2 amu')
                clean_val = re.sub(r"\s+[a-zA-Z\^\d/_~]+.*$", "", val)
                data[key] = float(clean_val)
            except ValueError:
                data[key] = val
    return data


def aggregate_pl_results():
    all_records = []

    print("Scanning calculation directories for properties...")

    for case in TARGET_CASES:
        case_dir = BASE_OUT_DIR / case

        for pipe in PIPELINES:
            prop_file = case_dir / pipe / "important_properties.txt"

            if prop_file.exists():
                print(f"Parsing: {case} -> {pipe}")
                properties = parse_properties_file(prop_file)

                # Create a baseline record identifier
                record = {"Functional_Setup": case, "Pipeline_Mode": pipe}

                # Merge parsed properties into the row record
                record.update(properties)
                all_records.append(record)
            else:
                print(f"Warning: Missing expected summary file at {prop_file}")

    if not all_records:
        print("No records found! Check if the calculation paths match.")
        return

    # Convert gathered list of dicts into a structured DataFrame
    df = pd.DataFrame(all_records)

    # Reorder columns dynamically to ensure identifiers stay on the left margin
    core_cols = ["Functional_Setup", "Pipeline_Mode"]
    other_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + other_cols]

    # Target output paths
    csv_out = BASE_OUT_DIR / "pl_summary_table.csv"
    md_out = BASE_OUT_DIR / "pl_summary_table.md"

    # Save outputs
    df.to_csv(csv_out, index=False)
    df.to_markdown(md_out, index=False)

    print(f"\nSuccessfully generated tables!")
    print(f"-> CSV Format: {csv_out}")
    print(f"-> Markdown Format (For README): {md_out}")


if __name__ == "__main__":
    aggregate_pl_results()