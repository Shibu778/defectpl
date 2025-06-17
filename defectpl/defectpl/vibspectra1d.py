import numpy as np
import math
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.style as style

## Use style file
style_file = Path(__file__).parent / "defectpl.mplstyle"
style.use(style_file)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.usetex"] = False


class VibrationalSpectra1D:
    """
    A class to calculate the vibrational structure of luminescence bands
    using a 1D model with different vibrational frequencies in ground and excited states.
    Python version of the fortran code by A. Alkauskas et al., Phys. Rev. Lett. 109, 267401 (2012)
    NN1, NN2 -number of vibrations to be included; avoid numbers NN1+NN2>80, the factorial too large.
    Default NN1 and NN2 are 22 and 52.
    Attributes:
        EZPL (float): Zero-phonon line energy (eV)
        w1 (float): Vibrational frequency in excited state (meV)
        w2 (float): Vibrational frequency in ground state (meV)
        DQ (float): Mass-weighted displacement (amu^1/2 * Angstrom)
        T (float): Temperature in K
        E0 (float): Start energy for spectrum (eV)
        dE (float): Energy step (eV)
        M (int): Number of energy points
        The output is produced for energies from E0 to (E0 + dE*M)
        For definitions of quantities see:  A. Alkauskas et al., Phys. Rev. Lett. 109, 267401 (2012)
    """

    def __init__(self, EZPL, w1, w2, DQ, T, E0, dE, M):
        # Input parameters
        self.EZPL = EZPL
        self.w1 = w1 / 1000.0  # Convert meV to eV
        self.w2 = w2 / 1000.0
        self.DQ = DQ
        self.T = T
        self.E0 = E0
        self.dE = dE
        self.M = int(M)

        # Constants
        self.PI = np.pi
        self.K2eV = 8.617e-5  ## k_b/e Kelvin to eV
        self.Factor = 15.46484755  ## (e*m_{amu})^1/2*1E-10/hbar
        self.Factor2 = 1.519262e15  ## e/hbar
        self.NN1, self.NN2 = 22, 52

        self.calculate_parameters()
        self.overlap_matrix = np.zeros((self.NN1 + 1, self.NN2 + 1))

    def factorial(self, k):
        return math.factorial(k)

    def hermite(self, n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return 2.0 * x
        y0, y1 = 1.0, 2.0 * x
        for k in range(2, n + 1):
            y0, y1 = y1, 2.0 * x * y1 - 2.0 * (k - 1) * y0
        return y1

    def overlap(self, m, n, rho, cosfi, sinfi):
        pr1 = (-1) ** n * np.sqrt(2 * cosfi * sinfi) * np.exp(-(rho**2))
        ix = 0.0
        k1, k2 = divmod(m, 2)
        l1, l2 = divmod(n, 2)

        for kx in range(k1 + 1):
            for lx in range(l1 + 1):
                k, l = 2 * kx + k2, 2 * lx + l2
                try:
                    pr2 = (
                        np.sqrt(float(self.factorial(n) * self.factorial(m)))
                        / (
                            self.factorial(k)
                            * self.factorial(l)
                            * self.factorial(k1 - kx)
                            * self.factorial(l1 - lx)
                        )
                        * 2.0 ** ((k + l - m - n) // 2)
                    )
                except ValueError:
                    pr2 = 0
                pr3 = (sinfi**k) * (cosfi**l)
                f = self.hermite(k + l, rho)
                ix += pr1 * pr2 * pr3 * f

        return ix

    def calculate_parameters(self):
        self.sigma = 0.70 * self.w2
        self.TE = self.T * self.K2eV
        self.w = self.w1 * self.w2 / (self.w1 + self.w2)
        self.rho = self.Factor * np.sqrt(self.w / 2.0) * self.DQ
        self.Erel1 = 0.5 * self.Factor**2 * self.w1**2 * self.DQ**2
        self.Erel2 = 0.5 * self.Factor**2 * self.w2**2 * self.DQ**2
        self.sinfi = np.sqrt(self.w2 / (self.w1 + self.w2))
        self.cosfi = np.sqrt(self.w1 / (self.w1 + self.w2))

        print(f"Relaxation energy in ground state: {self.Erel1:.6f} eV")
        print(f"Relaxation energy in excited state: {self.Erel2:.6f} eV")

    def compute_overlap_matrix(self):
        for i in range(self.NN1 + 1):
            for j in range(self.NN2 + 1):
                self.overlap_matrix[i, j] = self.overlap(
                    i, j, self.rho, self.cosfi, self.sinfi
                )

    def compute_spectrum(self):
        self.compute_overlap_matrix()
        Z = 1.0 / (1.0 - np.exp(-self.w1 / self.TE))
        Rpart = 0.0
        contr, en = [], []

        for i in range(self.NN1 + 1):
            weight = np.exp(-i * self.w1 / self.TE) / Z
            for j in range(self.NN2 + 1):
                val = self.overlap_matrix[i, j]
                contrib = weight * val**2
                Rpart += contrib
                contr.append(contrib)
                en.append(self.EZPL - j * self.w2 + i * self.w1)

        print(
            f"This number should be close to 1.00, if that is not the case, increase NN1 and NN2: {Rpart:.6f}"
        )

        self.overlap_data = {"contributions": contr, "energies": en}
        self.contr = contr
        self.en = en

    def compute_lineshape(self):
        self.compute_spectrum()
        dos = np.zeros(self.M)
        dosw3 = np.zeros(self.M)

        for l in range(self.M):
            E = self.E0 + l * self.dE
            for k, c in enumerate(self.contr):
                deltaE = self.en[k] - E
                dos[l] += (
                    c
                    * np.exp(-(deltaE**2) / (2 * self.sigma**2))
                    / (self.sigma * np.sqrt(2 * self.PI))
                )
            dosw3[l] = dos[l] * E**3

        dosw3 /= np.sum(dosw3) * self.dE

        self.spectral_data = {
            "energies": [self.E0 + l * self.dE for l in range(self.M)],
            "dos": dos.tolist(),
            "dosw3": dosw3.tolist(),
        }

    def save_results(
        self, overlap_file="overlap.json", lineshape_file="lineshape.json"
    ):
        with open(overlap_file, "w") as f:
            json.dump(self.overlap_data, f, indent=4)
        with open(lineshape_file, "w") as f:
            json.dump(self.spectral_data, f, indent=4)

    def plot_lineshape(self, save_file=None, figsize=(6, 4)):
        if not hasattr(self, "spectral_data"):
            raise ValueError("Run compute_lineshape() before plotting.")
        plt.figure(figsize=figsize)
        #        plt.plot(self.spectral_data["energies"], self.spectral_data["dos"], label='DOS', linestyle='--', linewidth=2)
        #        plt.plot(self.spectral_data["energies"], self.spectral_data["dosw3"], label='DOS * E³', linestyle='--', linewidth=2)
        plt.plot(
            self.spectral_data["energies"], self.spectral_data["dosw3"], linewidth=2
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (arb. units)")
        plt.title("Vibrational Lineshape")
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_file:
            plt.savefig(save_file, dpi=300)
            print(f"Lineshape plot saved to {save_file}")
        else:
            plt.show()

    def get_peak_position(self):
        """
        Returns the energy corresponding to the maximum of the normalized spectrum (DOS * E^3).
        """
        if not hasattr(self, "spectral_data"):
            raise ValueError("Run compute_lineshape() before accessing peak position.")
        dosw3 = np.array(self.spectral_data["dosw3"])
        energies = np.array(self.spectral_data["energies"])
        idx_max = np.argmax(dosw3)
        print(f"The peak position is: {energies[idx_max]:.3f} eV at {self.T} K.")
        return energies[idx_max], dosw3[idx_max]

    def get_fwhm(self):
        """
        Computes the full width at half maximum (FWHM) of the normalized spectrum (DOS * E^3).
        Returns: FWHM in eV.
        """
        if not hasattr(self, "spectral_data"):
            raise ValueError("Run compute_lineshape() before accessing FWHM.")
        dosw3 = np.array(self.spectral_data["dosw3"])
        energies = np.array(self.spectral_data["energies"])

        half_max = np.max(dosw3) / 2.0
        above_half = dosw3 >= half_max

        if not np.any(above_half):
            return 0.0

        indices = np.where(above_half)[0]
        fwhm = energies[indices[-1]] - energies[indices[0]]
        print(f"The full-width at half maximum (FWHM) is: {fwhm:.3f} eV at {self.T} K.")
        return fwhm
