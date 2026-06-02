"""
Useful functions for working with VASP output files, such as OUTCAR and vasprun.xml.
"""
import numpy as np
from pymatgen.io.vasp import Poscar, Outcar, Vasprun
from pymatgen.core import Structure
from typing import List, Tuple

def check_outcar_convergence(outcar_path: str) -> dict:
    """
    Checks whether a VASP calculation converged electronically and structurally.

    Scans the OUTCAR file to find structural convergence tokens and looks at 
    the final electronic step count to ensure it didn't stop because it hit 
    NELM (maximum electronic steps limit).

    Parameters
    ----------
    outcar_path : str
        The path to the VASP OUTCAR file.

    Returns
    -------
    dict
        A dictionary containing the convergence status results:
        - "structural_converged" (bool): True if ionic steps converged.
        - "electronic_converged" (bool): True if the final SCF cycle converged.
        - "finished_cleanly" (bool): True if VASP reached its normal end of run.
    """
    results = {
        "structural_converged": False,
        "electronic_converged": True, # Assumed True until proven stuck
        "finished_cleanly": False
    }

    # Open line-by-line. To preserve memory and maximize speed, we look for 
    # specific flags that dictate termination status.
    with open(outcar_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    # If the file is empty, it definitely didn't converge
    if not lines:
        return {k: False for k in results}

    # Scan the last ~100 lines for structural precision and termination flags
    tail_lines = lines[-100:]
    for line in tail_lines:
        if "reached required accuracy" in line:
            results["structural_converged"] = True
        if "User time" in list(line.split()[:2]) or "Total CPU time" in line:
            results["finished_cleanly"] = True

    # Scan the entire file for electronic step warnings.
    # VASP warns you if it hits max electronic cycles (e.g., NELM = 60) without converging.
    for line in lines:
        if "ELECTRONIC CONVERGENCE MINIMIZATION" in line and "not achieved" in line:
            results["electronic_converged"] = False
        # If it says "re-entering optimization" right after an un-converged warning,
        # it might converge on a later ionic step, so we continue checking.

    return results

def get_nions(outcar_path: str) -> int:
    """
    Extracts the number of ions (NIONS) from a VASP OUTCAR file.
    """
    with open(outcar_path, 'r', errors='ignore') as f:
        for line in f:
            if 'NIONS =' in line:
                return int(line.split('=')[-1])
                
    raise ValueError(f"Could not find 'NIONS =' in {outcar_path}")


def get_species_and_index_map(outcar_path: str) -> List[str]:
    """
    Extracts the atomic species list from an OUTCAR file to build an atom index map.
    
    Handles multi-occurrence or interleaved POTCAR definitions (e.g., N, C, N, C) 
    by aligning the element type sequence exactly with the length of the 
    'ions per type' array printed by VASP.
    """
    species_types = []
    ions_per_type = []

    with open(outcar_path, 'r', errors='ignore') as f:
        for line in f:
            # 1. Capture all POTCAR type lines sequentially as they appear
            if 'POTCAR:' in line and 'PAW' in line:
                tokens = line.split()
                if len(tokens) >= 3:
                    # Strip out valency extensions (e.g., 'Si_GW' -> 'Si')
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


def get_structures_and_forces(outcar_path: str, poscar_path: str = None) -> Tuple[List[Structure], List[np.ndarray]]:
    """
    Extracts all structures and forces from the OUTCAR file as a standalone function.

    Lattice matrices are dynamically updated per ionic configuration using the
    'VOLUME and BASIS-vectors are now :' blocks parsed strictly from the OUTCAR itself.
    Species tracking elements are parsed natively from the OUTCAR's POTCAR mappings.

    Parameters
    ----------
    outcar_path : str
        The path to the VASP OUTCAR file.
    poscar_path : str, optional
        Path to a POSCAR/CONTCAR to override chemical species symbols, by default None.
        If not provided, species mapping relies completely on the internal OUTCAR logic.

    Returns
    -------
    structures : list of pymatgen.core.Structure
        A list of Structure objects representing each ionic step with correct dynamic lattices.
    forces : list of np.ndarray
        A list of 2D NumPy arrays of shape (NIONS, 3) representing the total 
        forces (eV/Å) at each corresponding ionic step.
    """
    # 1. Dynamically get number of atoms
    natoms = get_nions(outcar_path)

    # 2. Parse species labels dynamically (Fallback to OUTCAR parser logic if poscar_path is None)
    if poscar_path:
        species = Poscar.from_file(poscar_path).structure.species
    else:
        species = get_species_and_index_map(outcar_path)

    # Initialize variables to keep track of state
    current_lattice = None
    structures = []
    forces = []

    with open(outcar_path, 'r', errors='ignore') as f:
        iterator = iter(f)
        
        for line in iterator:
            # 3. Dynamic Lattice Catching block: Parse from the basis vector summary block
            if "VOLUME and BASIS-vectors are now :" in line:
                # Advance 3 lines down to bypass:
                # -----------------------------------------------------------------------------
                #  energy-cutoff  :      520.00
                #  volume of cell :     1218.95
                next(iterator)
                next(iterator)
                next(iterator)  # This is the "direct lattice vectors   reciprocal..." header
                next(iterator)
                
                # Parse the next 3 lines for the 3x3 direct lattice vectors matrix
                lattice_matrix = []
                for _ in range(3):
                    # VASP prints direct elements first, then reciprocal elements on the same row.
                    # We extract only the first 3 float columns.
                    lattice_matrix.append([float(x) for x in next(iterator).split()[:3]])
                current_lattice = np.array(lattice_matrix)

            # 4. Atomic Position and Force Tracking Block
            if 'POSITION' in line and 'TOTAL-FORCE' in line:
                if current_lattice is None:
                    raise ValueError(
                        f"Parsed a POSITION block before finding a 'VOLUME and BASIS-vectors' block "
                        f"in {outcar_path}. Ensure the OUTCAR isn't corrupted."
                    )
                
                next(iterator)  # Skip the dashed border line '------'
                
                coords = np.zeros((natoms, 3))
                step_forces = np.zeros((natoms, 3))
                
                for i in range(natoms):
                    data = next(iterator).split()
                    coords[i] = [float(data[0]), float(data[1]), float(data[2])]
                    step_forces[i] = [float(data[3]), float(data[4]), float(data[5])]
                
                # Convert to complete pymatgen Structure object using the OUTCAR self-contained lattice
                struct = Structure(
                    lattice=current_lattice,
                    species=species,
                    coords=coords,
                    coords_are_cartesian=True
                )
                
                structures.append(struct)
                forces.append(step_forces)

    return structures, forces

class OutcarParser(Outcar):
    """
    A subclass of pymatgen's Outcar class that provides additional methods 
    for extracting specific information from the OUTCAR file.
    """
    def __init__(self, filename: str):
        super().__init__(filename)
        self.natoms = self.get_natoms()

    def get_natoms(self) -> int:
        """
        Extracts the number of atoms from the OUTCAR file using standalone utility.
        """
        self.natoms = get_nions(self.filename)
        return self.natoms

    def get_structures_and_forces(self, poscar_path: str = None):
        """
        Wrapper method calling the standalone utility function on self.filename.
        """
        return get_structures_and_forces(self.filename, poscar_path=poscar_path)

    def get_final_structure_and_forces(self, poscar_path: str = None):
        """
        Extracts only the final structure and forces from the OUTCAR file.
        """
        structures, forces = self.get_structures_and_forces(poscar_path=poscar_path)
        return structures[-1], forces[-1]
    
    def check_convergence(self) -> dict:
        """
        Wrapper method calling the standalone convergence check utility on self.filename.
        """
        return check_outcar_convergence(self.filename)
    

def get_final_structure_and_forces_from_outcar(outcar_path: str, poscar_path: str = None):
    """
    A standalone function to extract only the final structure and forces from the OUTCAR file.

    To get the species info POSCAR needs to be provided.
    """
    structures, forces = get_structures_and_forces(outcar_path, poscar_path=poscar_path)
    return structures[-1], forces[-1]



if __name__ == "__main__":
    # Example usage
    outcar_path = "../tests/data/hse06/gs/OUTCAR"
    structure, forces = get_final_structure_and_forces_from_outcar(outcar_path)
    print("Final Structure:", structure)
    print("Forces:", forces)