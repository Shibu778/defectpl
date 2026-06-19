#!/usr/bin/env python3
import os
from pathlib import Path

def main():
    # Base directory: C:\Users\shibu\local_code_library\high_throughput_framework\defectpl\examples\NV_diamond
    root_dir = Path(__file__).resolve().parent
    
    # Exact functional directory runs parsed from your file system list
    functional_runs = [
        "pbe_out",
        "frac_pbe_out",
        "hse06_out",
        "frac_hse06_out"
    ]
    
    # Order of analysis sub-pipelines matching your structural directory trees
    pipeline_modes = [
        "abs_gs_force_mode",
        "gs_zpl_disp_mode",
        "zpl_ems_force_mode"
    ]
    
    # Core plots available across your directories 
    primary_plots = [
        ("S_omega_vs_penergy.svg", "Spectral Function S(ω)"),
        ("intensity_vs_penergy.svg", "PL Intensity Spectrum"),
        ("HR_factor_vs_penergy.svg", "Huang-Rhys Factor Distribution"),
        ("penergy_vs_pmode.svg", "Phonon Energy vs Mode Index"),
        ("qk_vs_penergy.svg", "Partial delQ Displacements"),
        ("ipr_vs_penergy.svg", "Inverse Participation Ratio (IPR)"),
        ("loc_rat_vs_penergy.svg", "Localization Ratio Spectrum"),
        ("S_omega_HRf_ipr_vs_penergy.svg", "S(ω) / HRf / IPR Combined Plot")
    ]

    print("==> [START] Generating Markdown Quick-View Galleries for NV_diamond...")

    for run in functional_runs:
        run_dir = root_dir / run
        if not run_dir.exists():
            print(f"Skipping {run}: Directory does not exist layout-wise.")
            continue
            
        md_filename = f"gallery_{run}.md"
        md_path = root_dir / md_filename
        
        md_content = []
        md_content.append(f"# Plot Visualization Gallery: `{run}`")
        md_content.append(f"Automated quick-view gallery compiled for the `{run}` functional calculation pipeline.\n")
        md_content.append("---")
        
        # Add a localized Quick Navigation index
        md_content.append("### 📌 Quick Navigation")
        for pipeline in pipeline_modes:
            friendly_name = pipeline.replace("_", " ").title()
            anchor = pipeline.lower().replace("_", "-")
            md_content.append(f"- [{friendly_name}](#{anchor})")
        md_content.append("\n---\n")

        # Iterate through calculation pipeline modes
        for pipeline in pipeline_modes:
            pipeline_dir = run_dir / pipeline
            friendly_pipeline_name = pipeline.replace("_", " ").title()
            
            md_content.append(f"## 📊 {friendly_pipeline_name}")
            
            # Extract and drop in 'important_properties.txt' contents if present
            props_txt_path = pipeline_dir / "important_properties.txt"
            if props_txt_path.exists():
                md_content.append("<details><summary><b>📄 Show Run Metadata</b></summary>\n")
                md_content.append("```text")
                with open(props_txt_path, "r", encoding="utf-8") as txt_file:
                    md_content.append(txt_file.read().strip())
                md_content.append("```")
                md_content.append("</details>\n")
            
            # Generate HTML structured 2-column plot grid layout matching relative folder positions
            md_content.append("<table>")
            
            # Loop through primary plots 2 at a time for side-by-side rendering
            for i in range(0, len(primary_plots), 2):
                md_content.append("  <tr>")
                
                # Plot element 1 (Left column)
                filename_left, title_left = primary_plots[i]
                path_left = pipeline_dir / filename_left
                if path_left.exists():
                    rel_img_link = f"./{run}/{pipeline}/{filename_left}"
                    md_content.append(f'    <td align="center" width="50%"><b>{title_left}</b><br/><img src="{rel_img_link}" width="100%"/></td>')
                else:
                    md_content.append('    <td align="center" width="50%" style="color:gray;">Plot missing</td>')
                
                # Plot element 2 (Right column)
                if i + 1 < len(primary_plots):
                    filename_right, title_right = primary_plots[i+1]
                    path_right = pipeline_dir / filename_right
                    if path_right.exists():
                        rel_img_link = f"./{run}/{pipeline}/{filename_right}"
                        md_content.append(f'    <td align="center" width="50%"><b>{title_right}</b><br/><img src="{rel_img_link}" width="100%"/></td>')
                    else:
                        md_content.append('    <td align="center" width="50%" style="color:gray;">Plot missing</td>')
                else:
                    md_content.append('    <td width="50%"></td>')
                    
                md_content.append("  </tr>")
                
            md_content.append("</table>\n")
            md_content.append("---\n")
            
        # Write out Markdown file to disk
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
            
        print(f"[SUCCESS] Created quick-visualization viewer gallery: {md_filename}")

if __name__ == "__main__":
    main()