# -*- coding: utf-8 -*-
"""
Useful functions for working with VASP output files, such as OUTCAR, EIGENVAL, and vasprun.xml.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Poscar
from pymatgen.io.vasp.outputs import Eigenval
from pymatgen.electronic_structure.core import Spin


# =====================================================================
# Kohn-Sham Eigenvalue & Electronic Parsing Functions
# =====================================================================

def get_spin_multiplicity(homo_up_idx: int, homo_down_idx: int) -> float:
    """
    Calculate the spin multiplicity of the electronic configuration.

    Parameters
    ----------
    homo_up_idx : int
        The HOMO level position index extracted for the spin-up channel.
    homo_down_idx : int
        The HOMO level position index extracted for the spin-down channel.

    Returns
    -------
    float
        The calculated spin multiplicity ($2S + 1$).
    """
    S = abs(homo_up_idx - homo_down_idx) / 2.0
    return 2.0 * S + 1.0


def read_eigenval_file(filename: Union[str, Path], k_idx: int = 0) -> Dict[str, Any]:
    """
    Parse a VASP EIGENVAL file to extract spin-polarized eigenvalues at a k-point.

    Parameters
    ----------
    filename : str or pathlib.Path
        The destination track file path addressing the target EIGENVAL parameter.
    k_idx : int, default 0
        The absolute sequential list entry index picking the desired k-point grid item.

    Returns
    -------
    dict
        A data block payload containing keys mapping spin channels, index boundaries, 
        gaps, and calculation metrics.

    Raises
    ------
    ValueError
        If the input calculation structure data shows it is not spin-polarized.
    """
    # Inline import to avoid hard cyclical dependencies during runtime initialization
    from defectpl.ks_analysis import get_homo_lumo_idx

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
    data["selected_kpoint"] = [k_idx, list(eig.kpoints[k_idx])]
    data["spin_multiplicity"] = get_spin_multiplicity(data["homo_up_idx"], data["homo_down_idx"])
    
    return data


# =====================================================================
# Trajectory & Output Text File Parsing Functions
# =====================================================================

def check_outcar_convergence(outcar_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Checks whether a VASP calculation converged electronically and structurally.

    Scans the OUTCAR file to find structural convergence tokens and looks at 
    the final electronic step count to ensure it didn't stop because it hit 
    NELM (maximum electronic steps limit).

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.

    Returns
    -------
    dict
        A dictionary containing the convergence status results:
        
        - ``"structural_converged"`` (bool): True if ionic steps converged.
        - ``"electronic_converged"`` (bool): True if the final SCF cycle converged.
        - ``"finished_cleanly"`` (bool): True if VASP reached its normal end of run.

    Raises
    ------
    FileNotFoundError
        If the specified OUTCAR file does not exist.
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    results = {
        "structural_converged": False,
        "electronic_converged": True,  # Assumed True until proven stuck
        "finished_cleanly": False
    }

    # Use a deque to scan the tail without loading the entire file into memory
    tail_lines: deque = deque(maxlen=100)
    has_content = False

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            has_content = True
            tail_lines.append(line)
            # Scan the entire file sequentially for electronic warnings
            if "ELECTRONIC CONVERGENCE MINIMIZATION" in line and "not achieved" in line:
                results["electronic_converged"] = False

    if not has_content:
        return {k: False for k in results}

    # Evaluate the buffered tail lines for final statuses
    for line in tail_lines:
        if "reached required accuracy" in line:
            results["structural_converged"] = True
        
        tokens = line.split()
        if len(tokens) >= 2 and ("User" in tokens[0] and "time" in tokens[1]):
            results["finished_cleanly"] = True
        elif "Total CPU time" in line:
            results["finished_cleanly"] = True

    return results


def get_nions(outcar_path: Union[str, Path]) -> int:
    """
    Extracts the number of ions (NIONS) from a VASP OUTCAR file.

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.

    Returns
    -------
    int
        The total number of ions inside the VASP run.

    Raises
    ------
    FileNotFoundError
        If the specified OUTCAR file does not exist.
    ValueError
        If the 'NIONS =' string token cannot be found in the file structure.
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'NIONS =' in line:
                return int(line.split('=')[-1])
                
    raise ValueError(f"Could not find 'NIONS =' token within {outcar_path}")


def get_species_and_index_map(outcar_path: Union[str, Path]) -> List[str]:
    """
    Extracts the atomic species list from an OUTCAR file to build an atom index map.

    Handles multi-occurrence or interleaved POTCAR definitions (e.g., N, C, N, C) 
    by aligning the element type sequence exactly with the length of the 
    'ions per type' array printed by VASP.

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.

    Returns
    -------
    list of str
        A flat list of element symbols corresponding to each index in the simulation.

    Raises
    ------
    FileNotFoundError
        If the specified OUTCAR file does not exist.
    ValueError
        If species definitions or ion counts cannot be successfully determined 
        from the file stream.
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}")

    species_types = []
    ions_per_type = []

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1. Capture all POTCAR type lines sequentially as they appear
            if 'POTCAR:' in line and 'PAW' in line:
                tokens = line.split()
                if len(tokens) >= 3:
                    element = tokens[2].split('_')[0]
                    species_types.append(element)

            # 2. Capture how many ions exist for each type definition
            if 'ions per type =' in line:
                ions_per_type = [int(x) for x in line.split('=')[-1].split()]
                break

    if not species_types or not ions_per_type:
        raise ValueError(
            f"Could not fully parse species maps from {outcar_path}. "
            f"Found species types: {species_types}, Counts: {ions_per_type}"
        )
        
    if len(species_types) < len(ions_per_type):
        raise ValueError(
            f"OUTCAR processing error: Parsed fewer POTCAR species entries ({len(species_types)}) "
            f"than blocks specified in 'ions per type' ({len(ions_per_type)})."
        )

    # Slice the raw species array to match the actual number of active blocks
    species_types = species_types[:len(ions_per_type)]

    # Reconstruct the full index map sequence
    species_map = []
    for element, count in zip(species_types, ions_per_type):
        species_map.extend([element] * count)

    return species_map


def get_structures_and_forces(
    outcar_path: Union[str, Path], 
    poscar_path: Optional[Union[str, Path]] = None
) -> Tuple[List[Structure], List[np.ndarray]]:
    """
    Extracts all structures and forces from the OUTCAR file as a standalone function.

    Lattice matrices are dynamically updated per ionic configuration using the
    'VOLUME and BASIS-vectors are now :' blocks parsed strictly from the OUTCAR itself.
    Species tracking elements are parsed natively from the OUTCAR's POTCAR mappings.

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.
    poscar_path : str or pathlib.Path, optional
        Path to a POSCAR/CONTCAR to override chemical species symbols, by default None.
        If not provided, species mapping relies completely on the internal OUTCAR logic.

    Returns
    -------
    structures : list of pymatgen.core.Structure
        A list of Structure objects representing each ionic step with correct dynamic lattices.
    forces : list of numpy.ndarray
        A list of 2D NumPy arrays of shape (NIONS, 3) representing the total 
        forces (eV/Å) at each corresponding ionic step.

    Raises
    ------
    FileNotFoundError
        If either the outcar_path or poscar_path do not exist.
    ValueError
        If position indices are requested prior to matrix processing, or if 
        the data stream terminates prematurely.
    """
    outcar_path = Path(outcar_path)
    natoms = get_nions(outcar_path)

    if poscar_path:
        poscar_path = Path(poscar_path)
        if not poscar_path.is_file():
            raise FileNotFoundError(f"POSCAR reference file not found at {poscar_path}")
        species = Poscar.from_file(str(poscar_path)).structure.species
    else:
        species = get_species_and_index_map(outcar_path)

    current_lattice = None
    structures = []
    forces = []

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        iterator = iter(f)
        
        for line in iterator:
            try:
                # Catch dynamic lattice block
                if "VOLUME and BASIS-vectors are now :" in line:
                    for _ in range(4):
                        next(iterator)
                    
                    lattice_matrix = []
                    for _ in range(3):
                        lattice_matrix.append([float(x) for x in next(iterator).split()[:3]])
                    current_lattice = np.array(lattice_matrix)

                # Catch coordinate and trajectory forces data blocks
                if 'POSITION' in line and 'TOTAL-FORCE' in line:
                    if current_lattice is None:
                        raise ValueError(
                            f"Parsed a POSITION block before finding a lattice matrix "
                            f"in {outcar_path}. The file layout might be corrupted."
                        )
                    
                    next(iterator)  # Skip dashed line border
                    
                    coords = np.zeros((natoms, 3))
                    step_forces = np.zeros((natoms, 3))
                    
                    for i in range(natoms):
                        data = next(iterator).split()
                        coords[i] = [float(data[0]), float(data[1]), float(data[2])]
                        step_forces[i] = [float(data[3]), float(data[4]), float(data[5])]
                    
                    struct = Structure(
                        lattice=current_lattice,
                        species=species,
                        coords=coords,
                        coords_are_cartesian=True
                    )
                    structures.append(struct)
                    forces.append(step_forces)
                    
            except StopIteration:
                raise ValueError(f"Premature end of file encountered while parsing {outcar_path}")

    return structures, forces


class OutcarParser(Outcar):
    """
    A subclass of pymatgen's Outcar class providing clean extended query utilities.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path filename pointing to the targeting VASP OUTCAR run.
    """
    def __init__(self, filename: Union[str, Path]):
        filename_str = str(Path(filename).resolve())
        super().__init__(filename_str)
        self.filename_path = Path(filename_str)
        self.natoms = self.get_natoms()

    def get_natoms(self) -> int:
        """
        Extracts the total atom count from the underlying file.

        Returns
        -------
        int
            The quantity of ions configured within the VASP simulation.
        """
        self.natoms = get_nions(self.filename_path)
        return self.natoms

    def get_structures_and_forces(
        self, 
        poscar_path: Optional[Union[str, Path]] = None
    ) -> Tuple[List[Structure], List[np.ndarray]]:
        """
        Wrapper method pulling all configurations and corresponding forces.

        Parameters
        ----------
        poscar_path : str or pathlib.Path, optional
            Path to a POSCAR/CONTCAR to override elements, by default None.

        Returns
        -------
        structures : list of pymatgen.core.Structure
            A list of structures parsed sequentially from the run history.
        forces : list of numpy.ndarray
            Forces lists matching corresponding structural arrays.
        """
        return get_structures_and_forces(self.filename_path, poscar_path=poscar_path)

    def get_final_structure_and_forces(
        self, 
        poscar_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Structure, np.ndarray]:
        """
        Extracts only the final structural step and force values from the OUTCAR.

        Parameters
        ----------
        poscar_path : str or pathlib.Path, optional
            Path to a POSCAR/CONTCAR to override elements, by default None.

        Returns
        -------
        structure : pymatgen.core.Structure
            The last recorded dynamic chemical cell state.
        forces : numpy.ndarray
            The last recorded absolute atomic forces array.
        """
        structures, forces = self.get_structures_and_forces(poscar_path=poscar_path)
        return structures[-1], forces[-1]
    
    def check_convergence(self) -> Dict[str, bool]:
        """
        Executes standard convergence and processing sanity runs on the active file.

        Returns
        -------
        dict
            Status reporting flag values tracking internal convergence goals.
        """
        return check_outcar_convergence(self.filename_path)
    

def get_final_structure_and_forces_from_outcar(
    outcar_path: Union[str, Path], 
    poscar_path: Optional[Union[str, Path]] = None
) -> Tuple[Structure, np.ndarray]:
    """
    A standalone function to extract only the final structure and forces from the OUTCAR file.

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.
    poscar_path : str or pathlib.Path, optional
        Path to a POSCAR/CONTCAR to override elements, by default None.

    Returns
    -------
    structure : pymatgen.core.Structure
        The last recorded dynamic chemical cell state.
    forces : numpy.ndarray
        The last recorded absolute atomic forces array.
    """
    structures, forces = get_structures_and_forces(outcar_path, poscar_path=poscar_path)
    return structures[-1], forces[-1]


def get_first_structure_and_forces_from_outcar(
    outcar_path: Union[str, Path], 
    poscar_path: Optional[Union[str, Path]] = None
) -> Tuple[Structure, np.ndarray]:
    """
    A standalone function to extract only the first structure and forces from the OUTCAR file.

    Parameters
    ----------
    outcar_path : str or pathlib.Path
        The path to the VASP OUTCAR file.
    poscar_path : str or pathlib.Path, optional
        Path to a POSCAR/CONTCAR to override elements, by default None.

    Returns
    -------
    structure : pymatgen.core.Structure
        The first recorded dynamic chemical cell state.
    forces : numpy.ndarray
        The first recorded absolute atomic forces array.
    """
    structures, forces = get_structures_and_forces(outcar_path, poscar_path=poscar_path)
    return structures[0], forces[0]