# -*- coding: utf-8-*-
# Collection of functions for configuration coordinate diagram

# Note: This class is inspired from Nonrad package. The original Nonrad package is available at
# https://github.com/mturiansky/nonrad
# If you use this module, consider citing articles: https://doi.org/10.1103/PhysRevB.90.075202; https://doi.org/10.1016/j.cpc.2021.108056 

import os
import numpy as np
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple, Union, Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.style as style
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from scipy.optimize import curve_fit
from glob import glob
from matplotlib.ticker import MultipleLocator
from utils import AMU2KG, ANG2M, EV2J, HBAR_eVs

## Use style file
# style_file = Path(__file__).parent / "defectpl.mplstyle"
style_file = Path("./") / "defectpl.mplstyle"
style.use(style_file)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.usetex"] = False



class ConfigurationCoordinateDiagram:
    """Class for handling Configuration Coordinate Diagram calculations."""
    
    def __init__(self, ground_struct: Structure, excited_struct: Structure):
        """
        Initialize with ground and excited state structures.
        
        Args:
            ground_struct: Ground state structure
            excited_struct: Excited state structure
        """
        self.ground_struct = ground_struct
        self.excited_struct = excited_struct
        self.dQ = self._calculate_dQ()
        
    def _calculate_dQ(self) -> float:
        """Calculate dQ between ground and excited states."""
        return np.sqrt(np.sum(list(map(
            lambda x: x[0].distance(x[1])**2 * x[0].specie.atomic_mass,
            zip(self.ground_struct, self.excited_struct)
        ))))
    
    def generate_structures(self, displacements: np.ndarray, remove_zero: bool = True) -> Tuple[List[Structure], List[Structure]]:
        """
        Generate displaced structures for CCD.
        
        Args:
            displacements: Array of displacement values (e.g., np.linspace(-0.5, 0.5, 9))
            remove_zero: Whether to remove 0% displacement
            
        Returns:
            Tuple of (ground_structures, excited_structures)
        """
        displacements = np.array(displacements)
        if remove_zero:
            displacements = displacements[displacements != 0.]
            
        ground_structs = self.ground_struct.interpolate(self.excited_struct, nimages=displacements)
        excited_structs = self.ground_struct.interpolate(self.excited_struct, nimages=(displacements + 1.))
        return ground_structs, excited_structs
    
    def setup_calculations(self, displacements: np.ndarray, output_dir: Union[str, Path],
                          ground_input_dir: Union[str, Path], excited_input_dir: Union[str, Path],
                          input_files: List[str] = ['KPOINTS', 'POTCAR', 'INCAR']) -> None:
        """
        Set up calculation directories for CCD.
        
        Args:
            displacements: Array of displacement values
            output_dir: Base directory for output calculations
            ground_input_dir: Directory containing ground state input files
            excited_input_dir: Directory containing excited state input files
            input_files: List of input files to copy
        """
        output_dir = Path(output_dir)
        ground_input_dir = Path(ground_input_dir)
        excited_input_dir = Path(excited_input_dir)
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        ground_dir = output_dir / 'ground'
        excited_dir = output_dir / 'excited'
        os.makedirs(ground_dir, exist_ok=True)
        os.makedirs(excited_dir, exist_ok=True)
        
        # Generate structures
        ground_structs, excited_structs = self.generate_structures(displacements)
        
        # Write ground state calculations
        for i, struct in enumerate(ground_structs):
            working_dir = ground_dir / str(i)
            os.makedirs(working_dir, exist_ok=True)
            struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
            for f in input_files:
                copyfile(str(ground_input_dir / f), str(working_dir / f))
        
        # Write excited state calculations
        for i, struct in enumerate(excited_structs):
            working_dir = excited_dir / str(i)
            os.makedirs(working_dir, exist_ok=True)
            struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
            for f in input_files:
                copyfile(str(excited_input_dir / f), str(working_dir / f))
    
    @staticmethod
    def get_Q_from_struct(ground: Structure, excited: Structure, struct: Union[Structure, str],
                         tol: float = 1e-4, nround: int = 5) -> float:
        """
        Calculate Q value for a given structure.
        
        Args:
            ground: Ground state structure
            excited: Excited state structure
            struct: Structure or path to structure file
            tol: Distance cutoff for determining Q
            nround: Decimal places to round for Q determination
            
        Returns:
            Q value in amu^(1/2) Angstrom
        """
        if isinstance(struct, str):
            tstruct = Structure.from_file(struct)
        else:
            tstruct = struct
            
        dQ = np.sqrt(np.sum(list(map(
            lambda x: x[0].distance(x[1])**2 * x[0].specie.atomic_mass,
            zip(ground, excited)
        ))))
        
        excited_coords = excited.cart_coords
        ground_coords = ground.cart_coords
        struct_coords = tstruct.cart_coords
        
        dx = excited_coords - ground_coords
        ind = np.abs(dx) > tol
        
        poss_x = np.round((struct_coords - ground_coords)[ind] / dx[ind], nround)
        val, count = np.unique(poss_x, return_counts=True)
        
        return dQ * val[np.argmax(count)]
    
    def extract_PES(self, vasprun_paths: List[str], tol: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Potential Energy Surface from vasprun.xml files.
        
        Args:
            vasprun_paths: List of paths to vasprun.xml files
            tol: Tolerance for get_Q_from_struct
            
        Returns:
            Tuple of (Q values, energies)
        """
        num = len(vasprun_paths)
        Q, energy = (np.zeros(num), np.zeros(num))
        
        for i, vr_fname in enumerate(vasprun_paths):
            vr = Vasprun(vr_fname, parse_dos=False, parse_eigen=False)
            Q[i] = self.get_Q_from_struct(self.ground_struct, self.excited_struct, vr.structures[-1], tol=tol)
            energy[i] = vr.final_energy
            
        return Q, (energy - np.min(energy))
    
    @staticmethod
    def get_omega_from_PES(Q: np.ndarray, energy: np.ndarray, Q0: Optional[float] = None,
                          ax: Optional[Axes] = None, q: Optional[np.ndarray] = None) -> float:
        """
        Calculate harmonic phonon frequency from PES.
        
        Args:
            Q: Array of Q values
            energy: Array of energies
            Q0: Fix minimum of parabola
            ax: Matplotlib axis for plotting
            q: Q values for evaluating fit
            
        Returns:
            Harmonic phonon frequency in eV
        """
        def f(Q, omega, Q0, dE):
            return 0.5 * omega**2 * (Q - Q0)**2 + dE
            
        bounds = (-np.inf, np.inf) if Q0 is None else \
            ([-np.inf, Q0 - 1e-10, -np.inf], [np.inf, Q0 + 1e-10, np.inf])
        popt, _ = curve_fit(f, Q, energy, bounds=bounds)
        
        if ax is not None:
            q_L = np.max(Q) - np.min(Q)
            if q is None:
                q = np.linspace(np.min(Q) - 0.1 * q_L, np.max(Q) + 0.1 * q_L, 1000)
            ax.plot(q, f(q, *popt))
            
        return HBAR_eVs * popt[0] * np.sqrt(EV2J / (ANG2M**2 * AMU2KG))
    
    def analyze_ccd(self, ground_vaspruns: List[str], excited_vaspruns: List[str],
                   dE: float = 0.0, plot: bool = True, figsize: Tuple[int, int] = (5, 5),
                   xlim: Tuple[float, float] = (-3, 10), ylim: Tuple[float, float] = (-0.5, 4),
                   save_plot: Optional[str] = None) -> Tuple[float, float]:
        """
        Analyze CCD calculations and optionally plot results.
        
        Args:
            ground_vaspruns: List of ground state vasprun paths
            excited_vaspruns: List of excited state vasprun paths
            dE: Energy difference between minima
            plot: Whether to plot results
            figsize: Figure size
            xlim: X-axis limits
            ylim: Y-axis limits
            save_plot: Path to save plot (optional)
            
        Returns:
            Tuple of (ground_omega, excited_omega) in eV
        """
        # Extract PES
        Q_ground, E_ground = self.extract_PES(ground_vaspruns)
        Q_excited, E_excited = self.extract_PES(excited_vaspruns)
        E_excited = dE + E_excited
        
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(Q_ground, E_ground, s=10, label='Ground State')
            ax.scatter(Q_excited, E_excited, s=10, label='Excited State')
            
            # Fit curves
            q = np.linspace(xlim[0], xlim[1], 100)
            ground_omega = self.get_omega_from_PES(Q_ground, E_ground, ax=ax, q=q)
            excited_omega = self.get_omega_from_PES(Q_excited, E_excited, ax=ax, q=q)
            
            # Plot formatting
            ax.set_xlabel('$Q$ [amu$^{1/2}$ $\AA$]', fontsize=16)
            ax.set_ylabel('Energy [eV]', fontsize=16)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks(np.arange(xlim[0], xlim[1]+1, 2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.legend()
            
            if save_plot:
                plt.savefig(save_plot, bbox_inches="tight", dpi=400)
            plt.show()
        else:
            ground_omega = self.get_omega_from_PES(Q_ground, E_ground)
            excited_omega = self.get_omega_from_PES(Q_excited, E_excited)
            
        return ground_omega, excited_omega

    def estimate_vertical_transitions(
        self,
        ground_omega: float,
        excited_omega: float,
        dE: float,
        eps: float = 1e-6
    ) -> Tuple[float, float, float, float]:
        """
        Calculate vertical absorption and emission transition energies.
        Parameters
        ----------
        ground_omega : float
        Harmonic phonon frequency of the ground state (eV)
        excited_omega : float
        Harmonic phonon frequency of the excited state (eV)
        dQ : float
        Configuration coordinate offset between ground and excited states (amu^{1/2} Angstrom)
        dE : float
        Energy difference between minima of the two parabolas (eV)
        eps : float, optional
        Small number to avoid division by zero, by default 1e-6
        Returns
        -------
        tuple[float, float, float, float]
        Absorption energy, emission energy, FC factor (excited), FC factor (ground) in eV
        """
        if ground_omega < eps or excited_omega < eps:
            raise ValueError("Phonon frequencies must be positive")
        
        factor = self.dQ**2 * AMU2KG * ANG2M**2 / (HBAR_eVs**2) / EV2J
        # Franck-Condon factors
        FC_e = 0.5 * excited_omega**2 * factor
        FC_g = 0.5 * ground_omega**2 * factor
        # Vertical transitions
        E_abs = FC_e + dE
        E_em = dE - FC_g
        
        # Print results
        print(f"Absorption Energy: {E_abs:.6f} eV")
        print(f"Emission Energy:   {E_em:.6f} eV")
        print(f"Franck-Condon shift (excited): {FC_e:.6f} eV")
        print(f"Franck-Condon shift (ground):  {FC_g:.6f} eV")
        return E_abs, E_em, FC_e, FC_g