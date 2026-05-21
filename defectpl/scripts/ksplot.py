# Script to plot the KS eigenvalues near the bandgap
from pymatgen.io.vasp.outputs import Eigenval
from pymatgen.electronic_structure.core import Spin

import matplotlib.pyplot as plt
import numpy as np
import json

def get_homo_lumo_idx(eigenval, thr=0.6):
    """
    Find homo lumo position.

    Args:
        eigenval: A list of [energy, occupancy] pairs.
        thr: The threshold for considering a state as occupied.

    Returns:
        A tuple (homo_idx, lumo_idx) representing the indices of the HOMO and LUMO.

    """
    eigenval = np.array(eigenval)
    occupations = eigenval[:, 1]
    if len(set(occupations)) > 2:
        # If any other occupation than 1.0 and 0.0
        raise Warning(f"Fractional occupancies found. Threshold: {thr} is used.")

    homo_idx = max(i for i, occ in enumerate(occupations) if occ > thr)
    lumo_idx = min(i for i, occ in enumerate(occupations) if occ < thr)

    return homo_idx, lumo_idx

def get_spin_multiplicity(homo_up_idx, homo_down_idx):
    S = abs(homo_up_idx - homo_down_idx) / 2
    return 2 * S + 1

def read_eigenval_file(filename, k_idx=0):
    """
    Reads the eigenvalues from the EIGENVAL file and returns a dictionary with the eigenvalues for each spin channel.
    The k_idx parameter specifies which k-point to extract the eigenvalues from (default is 0, which corresponds to the first k-point).
    Assuming spin polarized calculation, the function will return a dictionary with keys "up" and "down" for the respective spin channels.
    """
    data = {}
    eig = Eigenval(filename, separate_spins=True)
    if eig.ispin != 2:
        raise ValueError("The calculation is not spin polarized.")
    print(f"Selecting the {k_idx}-th k-point from {eig.nkpt} k-points.")
    print(f"Selected k-point: {eig.kpoints[k_idx]}")
    data["up"] = list(eig.eigenvalues[Spin.up][k_idx])
    data["down"] = list(eig.eigenvalues[Spin.down][k_idx])
    data["homo_up_idx"], data["lumo_up_idx"] = get_homo_lumo_idx(data["up"])
    data["homo_down_idx"], data["lumo_down_idx"] = get_homo_lumo_idx(data["down"])
    data["homo_up"] = eig.eigenvalue_band_properties[2][0]
    data["homo_down"] = eig.eigenvalue_band_properties[2][1]
    data["lumo_up"] = eig.eigenvalue_band_properties[1][0]
    data["lumo_down"] = eig.eigenvalue_band_properties[1][1]
    data["hl_gap_up"] = eig.eigenvalue_band_properties[0][0]
    data["hl_gap_down"] = eig.eigenvalue_band_properties[0][1]
    data["nelect"] = eig.nelect
    data["nbands"] = eig.nbands
    data["nkpt"] = eig.nkpt
    data["selected_kpoint"] = [k_idx, eig.kpoints[k_idx]]
    data["spin_multiplicity"] = get_spin_multiplicity(data["homo_up_idx"], data["homo_down_idx"])
    return data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
def truncate_eigenval(eigenval, emin, emax):
    """
    Truncate the eigenvalues to a specified energy range.
    Args:
        eigenval: A list of eigenvalues e.g. [[e1, o1], [e2, o2], ...] pairs.
        emin: The minimum energy to include.
        emax: The maximum energy to include.
    """
    index = [i for i, (e, o) in enumerate(eigenval) if emin <= e <= emax]
    trunc_eig = [eigenval[i] for i in index]
    return trunc_eig, index

def find_degenerate_eigenvalues(eigenval, tol=1e-3):
    """
    Find degenerate eigenvalues within a specified tolerance.
    Args:
        eigenval: A list of eigenvalues e.g. [[e1, o1], [e2, o2], ...] pairs.
        tol: The energy tolerance for considering eigenvalues as degenerate.
    Returns:
        A list of lists, where each sublist contains the indices of degenerate eigenvalues.
    """
    degenerate_groups = []
    considered = []
    for i in range(len(eigenval)):
        if i in considered:
            continue
        group = [i]
        considered.append(i)
        for j in range(i + 1, len(eigenval)):
            if abs(eigenval[i][0] - eigenval[j][0]) < tol:
                group.append(j)
                considered.append(j)
        if len(group) >= 1:
            degenerate_groups.append(group)
    return degenerate_groups

def split_energy_occupation(eigenval):
    """
    Split the eigenvalues into separate lists of energies and occupations.
    Args:
        eigenval: A list of eigenvalues e.g. [[e1, o1], [e2, o2], ...] pairs.
    Returns:
        A tuple of two lists: (energies, occupations).
    """
    energies = [e for e, o in eigenval]
    occupations = [o for e, o in eigenval]
    return energies, occupations

def find_max_div_in_plot(degenrate_groups):
    """
    Find the maximum number of degenerate eigenvalues in any group, which can be used to adjust the plot spacing.
    Args:
        degenerate_groups: A list of lists, where each sublist contains the indices of degenerate eigenvalues.
    Returns:
        The maximum number of degenerate eigenvalues in any group.
    """
    max_div = max(len(group) for group in degenrate_groups)
    return max_div

def is_odd(n):
    return n % 2 == 1

def xpos_evaluation(npoint, max_div, sep=0.1, lim=10):
    """
    Calculate x values for plotting degenerate levels with a specified separation.
    Args:
        npoint: The number of degenerate levels to plot.
        max_div: The maximum number of degenerate levels in any group (used for normalization).
        sep: The separation between levels in the plot.
        lim: The maximum x limit.
    Returns:
        A list of x values for plotting the degenerate levels.

    Notes:
    lim = max_div * w + (max_div + 1) * sep
    w = (lim - sep) / max_div - sep
    midpos = 0.5 * lim
    """
    if npoint == 0:
        return [0]
    midpos = 0.5 * lim
    w = (lim - sep) / max_div - sep

    if npoint == 1:
        return [midpos]
    elif is_odd(npoint): # odd number of points
        xpos = [midpos]
        xpos += [midpos + i*w/2 + i*sep/2 for i in range(2, npoint, 2)]
        xpos += [midpos - i*w/2 - i*sep/2 for i in range(2, npoint, 2)]
        return xpos
    elif not is_odd(npoint): # even number of points
        xpos = [midpos + i*w/2 + i*sep/2 for i in range(1, npoint, 2)]
        xpos += [midpos - i*w/2 - i*sep/2 for i in range(1, npoint, 2)]
        return xpos
    
def get_x_values(deg_group, max_div, sep=0.1, lim=10):
    npoints = [len(group) for group in deg_group]
    xvalues = []
    for npoint in npoints:
        xvalues += xpos_evaluation(npoint, max_div, sep, lim)
    return xvalues

def negative_x_values(xvalues):
    return [-x for x in xvalues]

def get_occupied_unoccupied_split(occupations, xvalues, energies, threshold=0.6):
    occupied = {"xvalues": [], "energies": []}
    unoccupied = {"xvalues": [], "energies": []}
    for (occ, x, energy) in zip(occupations, xvalues, energies):
        if occ > threshold:
            occupied["xvalues"].append(x)
            occupied["energies"].append(energy)
        else:
            unoccupied["xvalues"].append(x)
            unoccupied["energies"].append(energy)
    return occupied, unoccupied

def extract_ksplot_data(eigenval_data, vbm, cbm, espan=1.0, sep=0.1, lim=10):
    """
    Extract the data needed for plotting the KS eigenvalues near the bandgap.
    """
    emin = vbm - espan
    emax = cbm + espan
    ks_plot_data = eigenval_data.copy()
    ks_plot_data["up"], up_idx = truncate_eigenval(eigenval_data["up"], emin, emax)
    ks_plot_data["down"], down_idx = truncate_eigenval(eigenval_data["down"], emin, emax)
    ks_plot_data["up_idx"] = up_idx
    ks_plot_data["down_idx"] = down_idx
    ks_plot_data["up_energies"], ks_plot_data["up_occupations"] = split_energy_occupation(ks_plot_data["up"])
    ks_plot_data["down_energies"], ks_plot_data["down_occupations"] = split_energy_occupation(ks_plot_data["down"])
    
    # Group degenerate eigenvalues for plotting purposes
    ks_plot_data["degenerate_up"] = find_degenerate_eigenvalues(ks_plot_data["up"])
    ks_plot_data["degenerate_down"] = find_degenerate_eigenvalues(ks_plot_data["down"])
    ks_plot_data["max_div_up"] = find_max_div_in_plot(ks_plot_data["degenerate_up"])
    ks_plot_data["max_div_down"] = find_max_div_in_plot(ks_plot_data["degenerate_down"])
    ks_plot_data["xvalues_up"] = negative_x_values(get_x_values(ks_plot_data["degenerate_up"], ks_plot_data["max_div_up"], sep, lim))
    ks_plot_data["xvalues_down"] = get_x_values(ks_plot_data["degenerate_down"], ks_plot_data["max_div_down"], sep, lim)

    # Occupied and unoccupied split
    ks_plot_data["occupied_up"], ks_plot_data["unoccupied_up"] = get_occupied_unoccupied_split(ks_plot_data["up_occupations"], ks_plot_data["xvalues_up"], ks_plot_data["up_energies"])
    ks_plot_data["occupied_down"], ks_plot_data["unoccupied_down"] = get_occupied_unoccupied_split(ks_plot_data["down_occupations"], ks_plot_data["xvalues_down"], ks_plot_data["down_energies"])

    # Basic information
    ks_plot_data["vbm"] = vbm
    ks_plot_data["cbm"] = cbm
    ks_plot_data["emin"] = emin
    ks_plot_data["emax"] = emax
    ks_plot_data["espan"] = espan
    ks_plot_data["sep"] = sep
    ks_plot_data["lim"] = lim
    ks_plot_data["w"] = (lim - sep) / min(ks_plot_data["max_div_up"], ks_plot_data["max_div_down"]) - sep
    return ks_plot_data

def initialize_plt_args(**kwargs):
    plt_args = {}

    return plt_args

def plot_spin_resolved_levels(data, style_file="ksplot_template.mplstyle", **kwargs):
    """
    Plotting the KS level given the required data.

    Args:
    data: A dictionary containing the necessary data for plotting, including:
        - "xvalues_up": A list of x values for the spin-up levels.
        - "up_energies": A list of energies for the spin-up levels.
        - "xvalues_down": A list of x values for the spin-down levels.
        - "down_energies": A list of energies for the spin-down levels.
        - "vbm": The energy of the valence band maximum (VBM).
        - "cbm": The energy of the conduction band minimum (CBM).
        - "emin": The minimum energy to display on the plot.
        - "emax": The maximum energy to display on the plot.
        - "lim": The maximum x limit for the plot.
    style_file: The path to a matplotlib style file for customizing the plot appearance.
    """
    plt.style.use(style_file)
    figsize=(6, 6)
    vbm_cbm_color = {"vbm": "orange", "cbm": "green", "alpha": 0.5}
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(-data["lim"], data["lim"])
    ax.set_ylim(data["emin"], data["emax"])
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    # ax.axhline(data["vbm"], color=vbm_cbm_color["vbm"], linestyle="--", label="VBM")
    ax.axhspan(data["emin"], data["vbm"], color=vbm_cbm_color["vbm"], alpha=vbm_cbm_color["alpha"])
    # ax.axhline(data["cbm"], color=vbm_cbm_color["cbm"], linestyle="--", label="CBM")
    ax.axhspan(data["cbm"], data["emax"], color=vbm_cbm_color["cbm"], alpha=vbm_cbm_color["alpha"])

    s = data["w"] * 200 * (figsize[0] / 6) # Adjust marker size based on the width of the plot and the number of degenerate levels    
    ax.scatter(data["xvalues_up"], data["up_energies"], color="black", marker="_", s=s)
    ax.scatter(data["xvalues_down"], data["down_energies"], color="black", marker="_", s=s)
    
    
    electron_markers = {"occupied": 'o', "unoccupied": 'x', "s": s /25}
    ax.scatter(data["occupied_up"]["xvalues"], data["occupied_up"]["energies"], color="k", marker=electron_markers["occupied"], s=electron_markers["s"])
    ax.scatter(data["unoccupied_up"]["xvalues"], data["unoccupied_up"]["energies"], color="k", marker=electron_markers["unoccupied"], s=electron_markers["s"])
    ax.scatter(data["occupied_down"]["xvalues"], data["occupied_down"]["energies"], color="k", marker=electron_markers["occupied"], s=electron_markers["s"])
    ax.scatter(data["unoccupied_down"]["xvalues"], data["unoccupied_down"]["energies"], color="k", marker=electron_markers["unoccupied"], s=electron_markers["s"])

    ax.set_ylabel("Energy (eV)")
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.savefig("ks_plot.png", dpi=300)
    

    
    
    


if __name__ == "__main__":
    vbm = 9.6747
    cbm = 13.7934
    espan = 1.0
    k_idx = 0
    eigenval_file = "../../EIGENVAL.gz"
    output_json = "ks_plot_data.json"
    eigenval_data = read_eigenval_file(eigenval_file, k_idx=k_idx)
    ks_plot_data = extract_ksplot_data(eigenval_data, vbm, cbm, espan=espan)
    write_json(ks_plot_data, output_json)
    print(f"KS plot data extracted and saved to {output_json}.")
    plot_spin_resolved_levels(ks_plot_data)