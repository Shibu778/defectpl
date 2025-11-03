# -*- coding: utf-8-*-
# Collection of functions for defectpl
# Author : Shibu Meher
# Date : 2024-10-23

# Note: This package is inspired from pyphotonics package. The original pyphotonics package is available at
# https://github.com/sheriftawfikabbas/pyphotonics
# Article: https://www.sciencedirect.com/science/article/pii/S0010465521003349?via%3Dihub

# For Theory Refer to A Alkauskas et al: https://iopscience.iop.org/article/10.1088/1367-2630/16/7/073026/meta

# Importing required libraries
import yaml
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.util.coord import pbc_shortest_vectors
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path
from defectpl.utils import *
from defectpl.io import read_properties
from phonopy.cui.load import load
from tabulate import tabulate
import json
import os
from defectpl.data import atom_data, symbol_map, isotope_data
from defectpl.plot import Plotter

## Use style file
style_file = Path(__file__).parent / "defectpl.mplstyle"
style.use(style_file)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.usetex"] = False


# Class for defectpl
class DefectPl:
    def __init__(
        self,
        band_yaml,
        contcar_gs,
        contcar_es,
        EZPL,
        gamma,
        resolution=1000,
        max_energy=5,
        sigma=6e-3,
        out_dir="./",
        plot_all=False,
        iplot_xlim=None,
        iylim=None,
        dump_data=True,
        max_freq=None,
        fig_format="pdf",
    ):
        """
        Initialize the class with the required parameters
        Parameters:
        =================
        band_yaml : str
            Path to the band.yaml file
        contcar_gs : str
            Path to the CONTCAR file for the ground state
        contcar_es : str
            Path to the CONTCAR file for the excited state
        EZPL : float
            Zero phonon line energy in eV
        gamma : float
            The gamma parameter representing the broadening of ZPL.
            The broadening has two contributions, homogeneous broadening
            due to anharmonic phonon interactions and the inhomogeneous
            broadening due to ensemble averaging.
            See : New J. Phys. 16 (2014) 073026
        resolution : int
            Number of points in the energy grid of 1 eV to calculate S(E)
        max_energy : float
            Maximum energy in eV for the energy grid to calculate S(E)
        sigma : float
            Standard deviation of the Gaussian broadening function
        out_dir : str
            Path to the output directory to save the output files
        plot_all : bool
            If True, all the plots will be generated. If False, no plots will be generated.
        iplot_xlim : list
            The x-axis limit for the intensity plot. Default is from
            ZPL-2000 to ZPL + 1000 meV. Give the range in meV.
        dump_data : bool
            If True, the data will be saved in the out_dir. If False, the data will not be saved.
        max_freq : float
            Maximum frequency in meV to plot the S(omega) vs omega plot.
        """
        self.band_yaml = band_yaml
        self.contcar_gs = contcar_gs
        self.contcar_es = contcar_es
        self.EZPL = EZPL
        self.gamma = gamma
        self.resolution = resolution
        self.max_energy = max_energy
        self.sigma = sigma
        self.out_dir = out_dir
        # data stores the information from the band.yaml file
        self.data = self.read_band_yaml_gamma(self.band_yaml)
        self.dR = self.calc_dR(self.contcar_gs, self.contcar_es)
        self.delR = self.calc_delR(self.dR)
        self.delQ = self.calc_delQ(self.mlist, self.dR)
        self.qks = self.calc_qks(self.mlist, self.dR, self.eigenvectors)
        self.Sks = self.calc_Sks(self.qks, self.frequencies)
        self.omega_range = [0, self.max_energy, self.max_energy * self.resolution]
        self.S_omega = self.calc_S_omega(self.frequencies, self.Sks, self.omega_range)
        self.HR_factor = self.calc_HR_factor(self.Sks)
        self.iprs = self.calc_IPR(self.eigenvectors)
        self.localization_ratio = self.calc_loc_rat(self.iprs, self.natoms)
        self.Sts = self.calc_St(self.S_omega)
        self.Gts = self.calc_Gts(self.Sts, self.HR_factor, self.gamma, self.resolution)
        self.A, self.I = self.calc_I(self.Gts, self.EZPL, self.resolution)
        if plot_all:
            if iplot_xlim is None:
                self.iplot_xlim = [(self.EZPL - 2) * EV2mEV, (self.EZPL + 1) * EV2mEV]
            else:
                self.iplot_xlim = iplot_xlim
            if max_freq:
                max_freq = max_freq / 1000
            try:
                self.plot_all(
                    out_dir=self.out_dir,
                    iplot_xlim=self.iplot_xlim,
                    max_freq=max_freq,
                    iylim=iylim,
                    fig_format=fig_format,
                )
            except Exception as e:
                print(f"Error in plotting: {e}")

        if dump_data:
            data = self.to_json(out_dir=self.out_dir)

    # Note: All the methods are written in such a way that
    # they can be used outside the class without depending on
    # the class variables. This is done to make the functions
    # more modular and reusable.
    def read_band_yaml(self, band_yaml):
        """
        Read the band.yaml file

        Parameters:
        =================
        band_yaml : str
            Path to the band.yaml file

        Returns:
        =================
        dict
            A dictionary with the data from the band.yaml file
        """
        with open(band_yaml, "r") as f:
            band_data = yaml.safe_load(f)

        return band_data

    def read_band_yaml_gamma(self, band_yaml):
        """
        This function reads the band.yaml file from phonopy.
        It is assumed that the calculation is done in gamma point only.
        Hence, we will only consider the bands at Gamma point.

        Parameters:
        =================
        band_yaml : str
            Path to the band.yaml file

        Returns:
        =================
        dict
            A dictionary with the following keys:
            - frequencies: a list of frequencies of the bands at Gamma point
            - eigenvectors: a list of eigenvectors of the bands at Gamma point
            - masses: a list of atomic masses of different species
            - natoms: number of atoms in the unit cell
            - nmodes: number of modes
            - nq: number of qpoints
        """
        self.band = self.read_band_yaml(band_yaml)
        q = 0  # Gamma point only
        self.natoms = self.band["natom"]
        self.nmodes = len(self.band["phonon"][q]["band"])
        self.nq = self.band["nqpoint"]
        gfrequencies = np.array(
            [self.band["phonon"][q]["band"][i]["frequency"] for i in range(self.nmodes)]
        )
        geigenvecs = np.array(
            [
                self.band["phonon"][q]["band"][i]["eigenvector"]
                for i in range(self.nmodes)
            ]
        )
        geigenvecs = geigenvecs[..., 0]
        self.mlist = [self.band["points"][i]["mass"] for i in range(self.natoms)]
        gfrequencies[gfrequencies < 0] = 0.0
        gfrequencies *= THZ2EV  # Convert to eV from THz from phonopy
        self.frequencies = gfrequencies
        self.eigenvectors = geigenvecs
        return {
            "frequencies": self.frequencies,
            "eigenvectors": self.eigenvectors,
            "masses": self.mlist,
            "natoms": self.natoms,
            "nmodes": self.nmodes,
            "nq": self.nq,
        }

    def calc_dR(self, constcar_gs, contcar_es):
        """This function calculates the difference in R between the excited state
        and ground state structures.

        Parameters:
        =================
        constcar_gs: str
            Path to the CONTCAR file of the ground state.
        contcar_es: str
            Path to the CONTCAR file of the excited state.

        Returns:
        =================
        dR : np.array
            The difference in the cartesian coordinates of the atoms between the
            ground state and the excited state. Excited state - Ground state. Units in Angstrom.
        """
        # Read the POSCAR files
        poscar_gs = Poscar.from_file(constcar_gs)
        poscar_es = Poscar.from_file(contcar_es)
        # Get the structures
        struct_gs = poscar_gs.structure
        struct_es = poscar_es.structure
        length = len(struct_gs)
        lattice = struct_gs.lattice
        # Calculate the dR
        dR = np.vstack(
            [
                pbc_shortest_vectors(
                    lattice, struct_gs.frac_coords[i], struct_es.frac_coords[i]
                )
                for i in range(length)
            ]
        ).reshape(length, 3)
        return dR

    def calc_delR(self, dR):
        r"""This function calculates the delta-R between given as follows:

        $$\Delta R = \sqrt{\sum_{\\alpha j}{} (R_{es}_{\alpha j} - R_{gs}_{\alpha j})^2}$$

        Parameters:
        =================
        dR: np.ndarray (natoms x 3)
            The difference in coordinate between the excited state and ground state structure.

        Returns:
        =================
        float
            The value of the delta-R.
        """
        return np.sqrt(np.sum(np.sum(dR**2)))

    def calc_delQ(self, mlist, dR):
        r"""This function calculates the delta-Q between given as follows:

        $$\Delta Q = \sqrt{\sum_{\alpha j}{} m_{\alpha}(R_{es}_{\alpha j} - R_{gs}_{\alpha j})^2}$$

        Parameters:
        =================
        mlist: list
            List of atomic masses in amu
        dR: np.array
            The dR between the excited state and ground state structure.

        Returns:
        =================
        float
            The value of the delta-Q.
        """
        mlist = np.array(mlist)
        return np.sqrt(np.sum(mlist * np.sum(dR**2, axis=1)))

    def calc_qk(self, mlist, dR, eigenvectors, k):
        """Calculates the qk value (Vibrational Displacement) corresponding
        to kth phonon mode.

        Parameters:
        =================
        mlist: list
            List of atomic masses of different species.
        dR: numpy array
            Difference in cartesian coordinate of atoms between the excited
            state and ground state structure.
        eigenvectors: numpy array
            Eigenvectors of the bands at Gamma point.
        k: int
            Index of the phonon mode.

        Returns:
        =================
        qk: float
            qk value corresponding to the phonon mode k
        """
        qk = 0
        for i in range(len(mlist)):
            qk += np.sqrt(mlist[i]) * np.dot(dR[i], eigenvectors[k][i])
        # convert to SI unit : Angstrom to meter, AMU to KG
        qk = qk * ANG2M * np.sqrt(AMU2KG)
        return qk

    def calc_qks(self, mlist, dR, eigenvectors):
        """Calculates the qk values (Vibrational Displacements) corresponding
        to each phonon mode.

        Parameters:
        =================
        mlist: np.array or list
            List of atomic masses of different species.
        dR: np.ndarray
            Difference in cart coords of atoms in the excited state and ground
            state structure.
        eigenvectors: np.ndarray
            Eigenvectors of the bands at Gamma point.

        Returns:
        =================
        qks: np.array
            qk values corresponding to each phonon mode.
        """
        qks = []
        for k in range(len(eigenvectors)):
            qk = 0
            for i in range(len(mlist)):
                qk += np.sqrt(mlist[i]) * np.dot(dR[i], eigenvectors[k][i])
            # convert to SI unit : Angstrom to meter, AMU to KG
            qk = qk * ANG2M * np.sqrt(AMU2KG)
            qks.append(qk)
        return np.array(qks)

    def calc_Sk(self, k, qk, frequencies):
        """Calculates the Sk value corresponding to kth phonon mode.

        Parameters:
        =================
        k: int
            Index of the phonon mode.
        qk: float
            qk value corresponding to the phonon mode.
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.

        Returns:
        =================
        Sk: float
            Sk value corresponding to the phonon mode.

        Note: HBAR_eVs is divided to convert the frequency from eV to Hz.
        """
        Sk = frequencies[k] * qk**2 / (2 * HBAR_Js * HBAR_eVs)
        return Sk

    def calc_Sks(self, qks, frequencies):
        """Calculates the partial HR factor for each phonon mode.

        Parameters:
        =================
        qks: np.array
            qk array corresponding to the phonon mode.
        frequencies: np.array
            List of frequencies of the bands at Gamma point. Frequency in eV.

        Returns:
        =================
        Sks: float
            Sk value corresponding to the phonon mode.
        """
        qks = np.array(qks)
        frequencies = np.array(frequencies)
        Sks = frequencies * qks**2 / (2 * HBAR_Js * HBAR_eVs)

        return Sks

    def gaussian(self, omega, omega_k, sigma):
        """This gaussian function is used to approximate the delta function.

        Parameters:
        =================
        omega: float or np.array
            The frequency at which the gaussian is evaluated.
        omega_k: float
            The frequency of the mode k. Mean of the gaussian.
        sigma: float
            The width of the gaussian.

        Returns:
        =================
        float or np.array
            The value of the gaussian at the frequency omega.
        """
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((omega - omega_k) / sigma) ** 2
        )

    def calc_S_omega(self, frequencies, Sks, omega_range, sigma=6e-3):
        """Calculate the S(omega) function.

        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        Sks: list
            List of Sk values corresponding to the phonon modes.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        sigma: float
            Width of the gaussian. Default is 6e-3 eV.

        Returns:
        =================
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        """
        S_omega = []
        omega = np.linspace(omega_range[0], omega_range[1], omega_range[2])
        for i in range(len(omega)):
            S_omega.append(
                np.sum(
                    [
                        Sks[j] * self.gaussian(omega[i], frequencies[j], sigma)
                        for j in range(len(frequencies))
                    ]
                )
            )

        return S_omega

    def calc_HR_factor(self, Sks):
        """Calculate the Huang-Rhys factor.

        Parameters:
        =================
        Sks: list
            List of Sk (partial HR factor) values corresponding to the phonon modes.

        Returns:
        =================
        HR_factor: float
            Huang-Rhys factor.
        """
        HR_factor = np.sum(Sks)
        return HR_factor

    def calc_IPR(self, eigenvectors):
        """Calculate the IPR (Inverse Participation Ratio) of phonon modes.

        Parameters:
        =================
        eigenvectors: numpy array
            Eigenvectors of the bands at Gamma point.

        Returns:
        =================
        IPRs: np.array
            Array of Inverse Participation Ratio for each phonon mode.
        """
        participations = np.array(
            [
                np.sum(eigenvectors[i] * eigenvectors[i], axis=1)
                for i in range(len(eigenvectors))
            ]
        )
        IPRs = np.sum(participations**2, axis=1)
        IPRs = 1 / IPRs
        return IPRs

    def calc_loc_rat(self, IPRs, nat):
        """Calculate the localization ratio of each phonon modes.

        Parameters:
        =================
        IPRs: np.array
            Array of Inverse Participation Ratio for each phonon mode.
        nat: int
            Number of atoms in the unit cell.

        Returns:
        =================
        localization_ratio: np.array
            Array of localization ratio for each phonon mode.
        """
        localization_ratio = nat / IPRs
        return localization_ratio

    def calc_St(self, S_omega):
        """
        Calculates the inverse discrete Fourier transform of S(omega) to get the time-domain signal.

        Parameters:
        =================
        S_omega: list
            List of S(omega) values. Here omega is in eV.

        Returns:
        =================
        St: list
            Time-domain signal.
        """
        Sts = np.fft.ifft(S_omega)
        Sts = np.fft.ifftshift(Sts)
        Sts = (
            2 * np.pi * Sts
        )  # np.fft.ifft gives 1/2pi factor so have to multiply by 2pi
        return Sts

    def calc_Gts(self, Sts, S, gamma, resolution):
        """Calculates the G(t) function.

        Parameters:
        =================
        Sts: list
            Time-domain signal.
        S: float
            St value corresponding to t=0. It also is the
            sum of the partial Hr factors Sks. It is the
            total HR factor.
        gamma: float
            ZPL broadening factor.
        resolution: float
            resolution of the time-domain signal.

        Returns:
        =================
        Gts: list
            G(t) function.
        """
        r = 1 / resolution
        G = np.exp(Sts - S)
        l = len(G)
        t = r * (np.array(range(l)) - l / 2)
        Gts = G * np.exp(-gamma * np.abs(t))
        return Gts

    def calc_I(self, Gts, EZPL, resolution):
        """
        Calculates the intensity of the spectrum.

        Parameters:
        =================
        Gts: list
            G(t) function.
        EZPL: float
            Zero Phonon Line energy.
        resolution: float
            resolution of the time-domain signal

        Returns:
        =================
        I: list
            Intensity of the spectrum.
        A: list
            Fourier transform of the G(t) function.
        """
        A = np.fft.fft(Gts)
        # Shifting the ZPL peak to the ZPL energy value
        A1 = A.copy()
        l = len(A)
        for i in range(l):
            A[(int(EZPL * resolution) - i) % l] = A1[i]

        omega_3 = (np.array(range(l)) / resolution) ** 3
        I = np.array(A) * np.array(omega_3)

        return A, I

    def to_json(self, out_dir):
        """Save all the properties to a json file.

        Parameters:
        =================
        out_dir: str
            Path to the output directory to save the json file.
        """
        out_path = Path(out_dir) / "properties.json"
        intensity = [(complex_num.real, complex_num.imag) for complex_num in self.I]
        # Use while reading intensity from json
        # self.I = np.array([complex(real, imag) for real, imag in I_list], dtype=np.complex128)
        data = {
            "class": "@DefectPl",
            "frequencies": self.frequencies.tolist(),
            "iprs": self.iprs.tolist(),
            "localization_ratio": self.localization_ratio.tolist(),
            "qks": self.qks.tolist(),
            "Sks": self.Sks.tolist(),
            "S_omega": self.S_omega,
            "omega_range": self.omega_range,
            "I": intensity,
            "resolution": self.resolution,
            "delta_R": self.delR.tolist(),
            "delta_Q": self.delQ,
            "HR_factor": self.HR_factor,
            "dR": self.dR.tolist(),
            "EZPL": self.EZPL,
            "gamma": self.gamma,
            "natoms": self.natoms,
            "masses": self.mlist,
            "max_energy": self.max_energy,
        }
        with open(out_path, "w") as f:
            json.dump(data, f)
        print("Properties are saved in a json file.")
        return data

    def plot_all(
        self,
        out_dir,
        iplot_xlim=None,
        max_freq=None,
        iylim=None,
        fig_format="pdf",
        **kwargs,
    ):
        """Plot all the properties.

        Parameters:
        =================
        out_dir: str
            Path to the output directory to save the plots.
        """
        # Plot phonon energy vs phonon mode index
        plotter = Plotter()
        plotter.plot_penergy_vs_pmode(
            frequencies=self.frequencies,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        # Plot IPR vs phonon energy
        plotter.plot_ipr_vs_penergy(
            self.frequencies,
            self.iprs,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        # Plot localization ratio vs phonon energy
        plotter.plot_loc_rat_vs_penergy(
            self.frequencies,
            self.localization_ratio,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )

        # Plot vibrational displacement vs phonon energy
        plotter.plot_qk_vs_penergy(
            self.frequencies,
            self.qks,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        # Plot partial HR factor vs phonon energy
        plotter.plot_HR_factor_vs_penergy(
            self.frequencies,
            self.Sks,
            plot=False,
            out_dir=out_dir,
            fig_format=fig_format,
        )
        # Plot S(omega) vs phonon energy
        plotter.plot_S_omega_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            plot=False,
            out_dir=out_dir,
            max_freq=max_freq,
            fig_format=fig_format,
        )
        # Plot S(omega) and Sks vs phonon energy
        plotter.plot_S_omega_Sks_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            plot=False,
            out_dir=out_dir,
            max_freq=max_freq,
            fig_format=fig_format,
        )
        # Plot S(omega) and Sks vs phonon energy
        plotter.plot_S_omega_Sks_Loc_rat_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.localization_ratio,
            plot=False,
            out_dir=out_dir,
            max_freq=max_freq,
            fig_format=fig_format,
        )
        # Plot S(omega), Sks and IPR vs phonon energy
        plotter.plot_S_omega_Sks_ipr_vs_penergy(
            self.frequencies,
            self.S_omega,
            self.omega_range,
            self.Sks,
            self.iprs,
            plot=False,
            out_dir=out_dir,
            max_freq=max_freq,
            fig_format=fig_format,
        )
        # Plot intensity vs photon energy
        plotter.plot_intensity_vs_penergy(
            self.frequencies,
            self.I,
            self.resolution,
            iplot_xlim,
            plot=False,
            out_dir=out_dir,
            iylim=iylim,
            fig_format=fig_format,
        )
        print("All plots are saved in the output directory.")


def mass_transformed_bandyaml(phonopy_yaml, force_sets_filename, masses, out_path="./"):
    """
    Calculate the frequency and eigenmodes of transformed phonon modes
    due to isotopic substitution.
    """
    band_yaml = Path(out_path) / "band.yaml"
    phonopy_params = Path(out_path) / "phonopy_params.yaml"
    phonon = load(phonopy_yaml, force_sets_filename=force_sets_filename)
    phonon.produce_force_constants()
    # phonon = load("phonopy_params.yaml")
    try:
        phonon.set_masses(masses)
    except:
        Exception("Something wrong with provided masses!!")
    phonon.save(filename=phonopy_params)
    phonon.run_band_structure([[[0, 0, 0]]], with_eigenvectors=True)
    phonon.write_yaml_band_structure(filename=band_yaml)
    return phonon


def get_mass_array(mass_dict, natom_dict, atom_seq):
    """
    Get the mass array from the mass and natom dictionary.
    """
    masses = []
    for key in atom_seq:
        masses.extend([mass_dict[key]] * natom_dict[key])
    return masses


def get_structure_info(filename):
    """
    Get the structure information from the POSCAR file.
    """
    poscar = Poscar.from_file(filename)
    structure = poscar.structure
    natoms = structure.num_sites
    natoms_dict = {
        key: int(value) for key, value in structure.composition.as_dict().items()
    }
    atom_seq = list(natoms_dict.keys())
    if sum(natoms_dict.values()) != natoms:
        raise Exception("Something wrong with atoms number in POSCAR file.")
    return natoms_dict, atom_seq


def get_isotope_info(species):
    """
    Get the isotope information for different species.
    """

    data = {}
    for sp in species:
        atomic_number = symbol_map[sp]
        atomic_mass = atom_data[atomic_number][3]
        print(f"Species: {atom_data[atomic_number][1]}({atom_data[atomic_number][2]})")
        print(f"Atomic Number: {atomic_number}")
        print(f"Mass: {atomic_mass}")
        print(f"Isotopes Information:")
        iso_data = isotope_data[sp]
        table = [["Isotope", "Mass", "Fraction"]]
        for iso in iso_data:
            table.append([iso[0], iso[1], iso[2]])

        print(tabulate(table, headers="firstrow", tablefmt="pretty"))
        print("\n")
        data[sp] = {"atomic_mass": atomic_mass, "iso_data": iso_data}
    return data


def get_atomic_mass_dict(species):
    """
    Get the atomic mass dictionary for the given species.
    """

    mass_dict = {}
    comp_dict = {}
    for sp in species:
        atomic_number = symbol_map[sp]
        atomic_mass = atom_data[atomic_number][3]
        mass_dict[sp] = atomic_mass
        comp_dict[sp] = round(atomic_mass)

    return mass_dict, comp_dict


def get_weighted_mass_dict(species, iso_data):
    """
    Get the weighted mass dictionary for the given species.

    This is similar to the given natural mass. The atomic mass
    is the weighted average of the isotopes in get_atomic_mass dict.
    So no need to use this function.
    """
    mass_dict = {}
    for sp in species:
        iso_data1 = iso_data[sp]["iso_data"]
        weighted_mass = 0
        for iso in iso_data1:
            weighted_mass += iso[1] * iso[2]
        mass_dict[sp] = weighted_mass
    return mass_dict


def generate_possible_indices(species):
    """
    Generate all possible isotopic compositions for the given species.
    """
    from itertools import product

    iso_data = get_isotope_info(species)
    all_indices = []
    for sp in species:
        iso = iso_data[sp]["iso_data"]
        all_indices.append(list(range(len(iso))))

    all_possible_indices = list(product(*all_indices))
    return all_possible_indices


def dict_to_comp_str(composition):
    """
    Convert the composition dictionary to a string.
    """
    comp_str = ""
    for key, value in composition.items():
        comp_str += f"{key}{value}"
    return comp_str


def get_composition(species, index):
    """
    Get the composition for the given index.
    """
    iso_data = get_isotope_info(species)
    composition = {}
    for i, sp in enumerate(species):
        iso_data1 = iso_data[sp]["iso_data"]
        iso = iso_data1[index[i]]
        composition[sp] = iso[0]
    return composition, dict_to_comp_str(composition)


def gen_all_possible_mass_dict(species):
    """
    Generate all possible mass dictionaries for the given species.
    """
    from itertools import product

    iso_data = get_isotope_info(species)
    all_mass_dict = {}
    mass_dict, composition = get_atomic_mass_dict(species)
    all_mass_dict["weighted_avg"] = {"composition": composition, "mass_dict": mass_dict}
    all_possible_indices = generate_possible_indices(species)
    for indices in all_possible_indices:
        mass_dict = {}
        for i, sp in enumerate(species):
            iso_data1 = iso_data[sp]["iso_data"]
            iso = iso_data1[indices[i]]
            mass_dict[sp] = iso[1]
        composition, comp_str = get_composition(species, indices)
        all_mass_dict[comp_str] = {"composition": composition, "mass_dict": mass_dict}
    return all_mass_dict


def get_all_mass_arrays(species, natoms_dict):
    """
    Get all mass arrays for the given species and natoms dictionary.
    """
    all_mass_dict = gen_all_possible_mass_dict(species)
    all_mass_arrays = {}
    for key, value in all_mass_dict.items():
        mass_dict = value["mass_dict"]
        mass_array = get_mass_array(mass_dict, natoms_dict, species)
        all_mass_arrays[key] = mass_array
    return all_mass_arrays


def poscar_to_masses(filename):
    """
    Get the masses from the POSCAR file.
    """
    natoms_dict, atom_seq = get_structure_info(filename)
    all_mass_arrays = get_all_mass_arrays(atom_seq, natoms_dict)
    return all_mass_arrays


def create_dir_structure(root_path, compositions):
    """
    Create the directory structure for the given compositions list.
    One folder for each composition is created in root_path.
    """
    for comp in compositions:
        comp_path = os.path.join(root_path, comp)
        if not os.path.exists(comp_path):
            os.makedirs(comp_path)
    print("Directory structure is created.")
