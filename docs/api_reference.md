Module defectpl.defectpl
========================

Classes
-------

`DefectPl(band_yaml, contcar_gs, contcar_es, EZPL, gamma, resolution=1000, max_energy=5, sigma=0.006, out_dir='./', plot_all=False, iplot_xlim=None)`


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


    ### Methods

    `calc_Gts(self, Sts, S, gamma, resolution)`
    :   Calculates the G(t) function.
        
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

    `calc_HR_factor(self, Sks)`
    :   Calculate the Huang-Rhys factor.
        
        Parameters:
        =================
        Sks: list
            List of Sk (partial HR factor) values corresponding to the phonon modes.
        
        Returns:
        =================
        HR_factor: float
            Huang-Rhys factor.

    `calc_I(self, Gts, EZPL, resolution)`
    :   Calculates the intensity of the spectrum.
        
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

    `calc_IPR(self, eigenvectors)`
    :   Calculate the IPR (Inverse Participation Ratio) of phonon modes.
        
        Parameters:
        =================
        eigenvectors: numpy array
            Eigenvectors of the bands at Gamma point.
        
        Returns:
        =================
        IPRs: np.array
            Array of Inverse Participation Ratio for each phonon mode.

    `calc_S_omega(self, frequencies, Sks, omega_range, sigma=0.006)`
    :   Calculate the S(omega) function.
        
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

    `calc_Sk(self, k, qk, frequencies)`
    :   Calculates the Sk value corresponding to kth phonon mode.
        
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

    `calc_Sks(self, qks, frequencies)`
    :   Calculates the partial HR factor for each phonon mode.
        
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

    `calc_St(self, S_omega)`
    :   Calculates the inverse discrete Fourier transform of S(omega) to get the time-domain signal.
        
        Parameters:
        =================
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        
        Returns:
        =================
        St: list
            Time-domain signal.

    `calc_dR(self, constcar_gs, contcar_es)`
    :   This function calculates the difference in R between the excited state
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

    `calc_delQ(self, mlist, dR)`
    :   This function calculates the delta-Q between given as follows:
        
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

    `calc_delR(self, dR)`
    :   This function calculates the delta-R between given as follows:
        
        $$\Delta R = \sqrt{\sum_{\\alpha j}{} (R_{es}_{\alpha j} - R_{gs}_{\alpha j})^2}$$
        
        Parameters:
        =================
        dR: np.ndarray (natoms x 3)
            The difference in coordinate between the excited state and ground state structure.
        
        Returns:
        =================
        float
            The value of the delta-R.

    `calc_loc_rat(self, IPRs, nat)`
    :   Calculate the localization ratio of each phonon modes.
        
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

    `calc_qk(self, mlist, dR, eigenvectors, k)`
    :   Calculates the qk value (Vibrational Displacement) corresponding
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

    `calc_qks(self, mlist, dR, eigenvectors)`
    :   Calculates the qk values (Vibrational Displacements) corresponding
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

    `gaussian(self, omega, omega_k, sigma)`
    :   This gaussian function is used to approximate the delta function.
        
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

    `plot_HR_factor_vs_penergy(self, frequencies, Sks, plot=False, out_dir='./', file_name='HR_factor_vs_penergy.pdf')`
    :   Plot the partial HR factor vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        Sks: list
            List of partial HR factor values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "HR_factor_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_S_omega_Sks_Loc_rat_vs_penergy(self, frequencies, S_omega, omega_range, Sks, localization_ratio, plot=False, out_dir='./', file_name='S_omega_HRf_loc_rat_vs_penergy.pdf')`
    :   Plot the S(omega), partial HR factor and localization ratio vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        localization_ratio: list
            List of localization ratio values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_HRf_loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_S_omega_Sks_ipr_vs_penergy(self, frequencies, S_omega, omega_range, Sks, iprs, plot=False, out_dir='./', file_name='S_omega_HRf_ipr_vs_penergy.pdf')`
    :   Plot the S(omega), partial HR factor and IPR vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        iprs: list
            List of IPR values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_HRf_loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_S_omega_Sks_vs_penergy(self, frequencies, S_omega, omega_range, Sks, plot=False, out_dir='./', file_name='S_omega_vs_penergy.pdf')`
    :   Plot the S(omega) vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        Sks: list
            List of partial HR factor values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_S_omega_vs_penergy(self, frequencies, S_omega, omega_range, plot=False, out_dir='./', file_name='S_omega_vs_penergy.pdf')`
    :   Plot the S(omega) vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        S_omega: list
            List of S(omega) values. Here omega is in eV.
        omega_range: list
            Range of omega values. [Start, End, Number of points]
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "S_omega_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_all(self, out_dir, iplot_xlim=None)`
    :   Plot all the properties.
        
        Parameters:
        =================
        out_dir: str
            Path to the output directory to save the plots.

    `plot_intensity_vs_penergy(self, frequencies, I, resolution, xlim, plot=False, out_dir='./', file_name='intensity_vs_penergy.pdf')`
    :   Plot the intensity vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        I: np.array of complex values
            List of intensity values for each phonon mode.
        resolution: float
            Resolution of the time-domain signal.
        xlim: list
            Range of phonon energy values. [Start, End]. Unit meV.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "intensity_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_ipr_vs_penergy(self, frequencies, iprs, plot=False, out_dir='./', file_name='ipr_vs_penergy.pdf')`
    :   Plot the IPR vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        iprs: list
            List of IPR values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "ipr_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_loc_rat_vs_penergy(self, frequencies, localization_ratio, plot=False, out_dir='./', file_name='loc_rat_vs_penergy.pdf')`
    :   Plot the localization ratio vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        localization_ratio: list
            List of Localization ratio values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "loc_rat_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_penergy_vs_pmode(self, frequencies, plot=False, out_dir='./', file_name='penergy_vs_pmode.pdf')`
    :   Plot the phonon energy vs phonon mode index.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "penergy_vs_pmode.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `plot_qk_vs_penergy(self, frequencies, qks, plot=False, out_dir='./', file_name='qk_vs_penergy.pdf')`
    :   Plot the vibrational displacement vs phonon energy.
        
        Parameters:
        =================
        frequencies: list
            List of frequencies of the bands at Gamma point. Frequency in eV.
        qks: list
            List of vibrational displacement values for each phonon mode.
        plot: bool
            If True, the plot will be shown. If False, the plot will be saved. Default is False.
        out_dir: str
            Path to the output directory to save the plot. Default is "./".
        file_name: str
            Name of the file to save the plot. Default is "qk_vs_penergy.pdf". If the format is
            not specified or not valid, it will be saved in pdf format.

    `read_band_yaml(self, band_yaml)`
    :   Read the band.yaml file
        
        Parameters:
        =================
        band_yaml : str
            Path to the band.yaml file
        
        Returns:
        =================
        dict
            A dictionary with the data from the band.yaml file

    `read_band_yaml_gamma(self, band_yaml)`
    :   This function reads the band.yaml file from phonopy.
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