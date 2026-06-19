# Photoluminescence Lineshape Calculation: A Theoretical Tutorial
**Framework Implementation Guide for `defectpl`**

This tutorial presents the quantum mechanical and atomistic framework used to compute the photoluminescence (PL) lineshape of localized defects in semiconductors and insulators from first principles. The equations, approximations, and algorithms below follow the formalisms established by Alkauskas *et al.* (2014) [1, 2] and establish the mathematical basis for the `defectpl` package.

---

## 1. Core Physical Approximations

To simulate the optical response of deep-level defects under an electronic transition from an initial excited state ($e$) to a final ground state ($g$), three baseline physical approximations are conventionally applied:

1. **The Franck-Condon Approximation:** The transition dipole moment $\vec{\mu}_{eg}$ between the initial and final electronic states is assumed to depend weakly on the nuclear coordinates $\mathbf{R}$. It is treated as a constant evaluated at the equilibrium configuration [1].
2. **The Linear Electron-Phonon Coupling Model:** The electronic energy depends linearly on atomic displacements around the equilibrium configuration. Complex non-linearities such as the dynamic Jahn-Teller (JT) effect are neglected when resolving the broad phonon sideband [1].
3. **The Harmonic Approximation:** The nuclear potential energy surfaces of both the ground and excited states are assumed to be harmonic [1, 2].

Under these definitions, the absolute luminescence intensity $I(\hbar\omega)$ (representing photons emitted per unit time per unit energy) at temperature $T$ is governed by Fermi's Golden Rule [1]:

$$I(\hbar\omega) = \frac{n_{D}\omega^{3}}{3\epsilon_{0}\pi c^{3}\hbar}|\vec{\mu}_{eg}|^{2} \sum_{m,n} w_{n} |\langle\chi_{gm}|\chi_{en}\rangle|^{2} \delta(E_{\text{ZPL}} + E_{en} - E_{gm} - \hbar\omega)$$

Where:
* $n_{D}$ is the refractive index of the host material [1].
* $w_{n}$ is the statistical Boltzmann weight of the initial vibrational state $n$ in the electronic excited state manifold [1, 2].
* $\chi_{en}$ and $\chi_{gm}$ represent the vibrational states of the excited and ground electronic states, respectively [1, 2].
* $E_{\text{ZPL}}$ is the Zero-Phonon Line energy [1].
* $E_{en}$ and $E_{gm}$ represent the discrete vibrational energies within their respective manifolds [1, 2].

The $\omega^{3}$ prefactor stems directly from the photon density of states ($\sim\omega^{2}$) scaled by the electric field perturbation energy ($|\vec{E}|^{2}\sim\omega$) [1]. Because measuring absolute intensity is experimentally challenging, `defectpl` models the normalized luminescence intensity $L(\hbar\omega)$ [1]:

$$L(\hbar\omega) = C \omega^{3} A(\hbar\omega)$$

where $C$ is a normalization constant ($C^{-1} = \int A(\hbar\omega)\omega^{3} d(\hbar\omega)$), and $A(\hbar\omega)$ is the fundamental **optical spectral function** containing the vibrational overlap integrals [1]:

$$A(\hbar\omega) = \sum_{m,n} w_{n} |\langle\chi_{gm}|\chi_{en}\rangle|^{2} \delta(E_{\text{ZPL}} + E_{en} - E_{gm} - \hbar\omega)$$

---

## 2. Multi-Mode Formalism (Identical Hessian Approximation)

When expanding to a full multi-mode treatment involving thousands of crystal lattice degrees of freedom, the **identical Hessian approximation** ($\omega_{k}^{(e)} \approx \omega_{k}^{(g)} \equiv \omega_{k}$) is introduced to make the problem computationally tractable [1]. At $T = 0\text{ K}$, all initial population resides in the lowest vibrational state ($n=0$), mapping the optical spectral function to the **spectral function of electron-phonon coupling** $S(\hbar\omega)$ [1]:

$$S(\hbar\omega) = \sum_{k} S_{k} \delta(\hbar\omega - \hbar\omega_{k})$$

where $S_{k} = \frac{\omega_{k} q_{k}^{2}}{2\hbar}$ represents the partial Huang-Rhys (HR) factor for a discrete phonon mode $k$ [1]. The variable $q_{k}$ is the mode-displaced configuration coordinate. The `defectpl` package supports two equivalent mathematical paths to compute $q_{k}$ based on the raw data available from first-principles calculations:

### Path A: Displacement Mode
When the explicit equilibrium atomic coordinates of both the excited state ($\mathbf{R}_{e}$) and ground state ($\mathbf{R}_{g}$) are known, $q_{k}$ is evaluated via structural differences [1]:

$$q_{k} = \sum_{a, i} m_{a}^{1/2} \left(R_{e;a,i} - R_{g;a,i}\right) \Delta r_{k;a,i}$$

Here, $a$ indexes the atoms, $i \in \{x, y, z\}$ represents the Cartesian directions, $m_{a}$ is the atomic mass, and $\Delta r_{k;a,i}$ is the normalized displacement vector of atom $a$ along direction $i$ in phonon mode $k$ (the eigenvectors of the mass-weighted Hessian matrix) [1].

### Path B: Force Mode
Alternatively, the coordinate translation can be expressed via differences in Hellmann-Feynman forces acting on fixed atomic sites [1]:

$$q_{k} = \frac{1}{\omega_{k}^{2}} \sum_{a, i} \frac{1}{m_{a}^{1/2}} \left(F_{e;a,i} - F_{g;a,i}\right) \Delta r_{k;a,i}$$

where $F_{e;a,i} - F_{g;a,i}$ represents the change in forces when the electronic state shifts instantly at a fixed spatial geometry [1]. This is equivalent to the displacement formulation under a strict harmonic potential since $(\vec{R}_{e} - \vec{R}_{g}) = -\hat{H}^{-1}(\vec{F}_{e} - \vec{F}_{g})$, where $\hat{H}$ is the mass-unweighted Hessian [1].

*Note: In `defectpl`, when evaluating an engine state running in Force Mode, spatial displacement scalars like $\Delta Q$ and $\Delta R$ are omitted during diagnostic text generation to preserve mathematical consistency with the force-derived fields.*

Integrating the entire spectral density function yields the **total Huang-Rhys factor** ($S$) for the optical transition [1]:

$$S = \int_{0}^{\infty} S(\hbar\omega) d(\hbar\omega) = \sum_{k} S_{k}$$

---

## 3. The Generating Function Approach

Rather than calculating explicitly convoluted multi-phonon combination states term by term, `defectpl` resolves $A(\hbar\omega)$ using the time-dependent generating function approach developed by Lax, Kubo, and Toyozawa [1]. 

The optical spectral function is calculated via the Fourier transform of a time-domain generating function $G(t)$ [1]:

$$A(E_{\text{ZPL}} - \hbar\omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} G(t) e^{i\omega t - \gamma |t|} dt$$

where $\gamma$ is a semi-empirical broadening parameter applied to reproduce the finite experimental width of the Zero-Phonon Line (arising from homogeneous thermal interactions or inhomogeneous sample strain) [1]. 

The generating function $G(t)$ is computed directly from the electron-phonon coupling spectrum [1]:

$$G(t) = e^{S(t) - S(0)}$$

where $S(t)$ is the Fourier transform of the spectral density [1]:

$$S(t) = \int_{0}^{\infty} S(\hbar\omega) e^{-i\omega t} d(\hbar\omega)$$

and $S(0) \equiv S = \int_{0}^{\infty} S(\hbar\omega) d(\hbar\omega)$ is the total HR factor [1].

### Physical Metrics Derived from the Sideband
* **Debye-Waller Factor ($w_{\text{ZPL}}$):** Represents the fractional intensity or weight contained strictly within the Zero-Phonon Line relative to the total emission profile [1]. Under identical ground and excited state potentials, it evaluates to [1]:
  $$w_{\text{ZPL}} = e^{-S}$$
* **Experimental Apparent HR Factor ($\tilde{S}$):** If deduced directly from the log-weight of the ZPL in a recorded spectrum, $\tilde{S} = -\ln(w_{\text{ZPL}})$ [1]. Because the real physical emission intensity $L(\hbar\omega)$ is skewed toward higher energies by the $\omega^{3}$ prefactor compared to the bare spectral shape $A(\hbar\omega)$, $\tilde{S}$ typically underestimates the true total theoretical coupling factor $S$ ($\tilde{S} < S$) [1].

---

## 4. 1D Harmonic Approximation Lineshape

For simple validation or handling localized systems dominated by a single effective vibrational mode, a 1D harmonic model can simulate lineshapes. This is implemented in the `VibrationalSpectra1D` class of `defectpl`. This model abandons the identical-Hessian restriction, explicitly allowing the effective phonon frequency of the ground state ($\omega_{1}$) to differ from that of the excited state ($\omega_{2}$) [1, 2].

### A. Configuration Coordinate Parameters and Unit Conversions
The calculation begins with an input mass-weighted configuration coordinate offset $\Delta Q$, expressed in units of $\text{amu}^{1/2}\cdot\text{Å}$ [1, 2]. To translate this value into a dimensionless displacement parameter $\rho$ appropriate for evaluating quantum mechanical harmonic oscillator wavefunctions, a unified conversion factor ($\text{FACTOR}$) is established using standard SI constants [2]:

$$\text{FACTOR} = \frac{\sqrt{\text{AMU}_{kG}} \cdot \text{ANG}_{M}}{\hbar_{J\cdot S}} = \frac{\sqrt{1.66053906 \times 10^{-27}\text{ kg}} \cdot 10^{-10}\text{ m}}{1.054571817 \times 10^{-34}\text{ J}\cdot\text{s}} \approx 15.46484755$$

Using an effective reduced frequency $\omega$ determined by both electronic states:

$$\omega = \frac{\omega_{1}\omega_{2}}{\omega_{1} + \omega_{2}}$$

the dimensionless displacement parameter $\rho$ governing the structural offset is computed as [2]:

$$\rho = \text{FACTOR} \cdot \sqrt{\frac{\omega}{2}} \cdot \Delta Q$$

The vertical relaxation energies (Franck-Condon shifts) associated with the ground state ($E_{\text{rel},1}$) and the excited state ($E_{\text{rel},2}$) are derived quadratically [2]:

$$E_{\text{rel},1} = \frac{1}{2} (\text{FACTOR})^{2} \cdot \omega_{1}^{2} \cdot (\Delta Q)^{2}$$

$$E_{\text{rel},2} = \frac{1}{2} (\text{FACTOR})^{2} \cdot \omega_{2}^{2} \cdot (\Delta Q)^{2}$$

### B. Franck-Condon Matrix Elements with Frequency Distortion
Because $\omega_{1} \neq \omega_{2}$, the vibrational wavefunctions are both translated in space and altered in width (distorted) [2]. The mixing of states is parameterized by rotation-like parameters $\cos\phi$ and $\sin\phi$ defined via the frequencies [2]:

$$\cos\phi = \sqrt{\frac{\omega_{1}}{\omega_{1} + \omega_{2}}}, \quad \sin\phi = \sqrt{\frac{\omega_{2}}{\omega_{1} + \omega_{2}}}$$

The transition overlap matrix element mapping the $i$-th vibrational level of the initial excited state to the $j$-th vibrational level of the final ground state, $M_{i,j} = \langle\chi_{g,j}|\chi_{e,i}\rangle$, is evaluated sequentially. For the zero-zero transition ($i=0, j=0$), the baseline overlap equals [2]:

$$M_{0,0} = \sqrt{2\cos\phi\sin\phi} \cdot \exp\left(-\frac{\rho^{2}}{2}\right)$$

Higher-order matrix elements are computed dynamically through explicit coordinate transformations or corresponding Hermite polynomial relations implemented in `utils.calculate_overlap_element`, referencing the initial structural state displacement variables [2].

### C. Thermal Distribution and Lineshape Convolution
At a finite temperature $T$, the initial excited-state vibrational levels are populated according to a grand canonical Boltzmann distribution. The partition function $Z$ and the statistical weights $W_{i}$ for the excited state are quantified via [2]:

$$Z = \frac{1}{1 - \exp\left(-\frac{\omega_{1}}{k_{B}T}\right)}, \quad W_{i} = \frac{1}{Z} \exp\left(-\frac{i \cdot \omega_{1}}{k_{B}T}\right)$$

The discrete contribution $C_{i,j}$ and corresponding transition energy $E_{i,j}$ for each pair are resolved across predefined limits (up to $i = N_{1}$ and $j = N_{2}$) [1, 2]:

$$C_{i,j} = W_{i} \cdot |M_{i,j}|^{2}, \quad E_{i,j} = E_{\text{ZPL}} - j\omega_{2} + i\omega_{1}$$

The code verifies the closure relation of the calculated manifold matrix space by ensuring the total probability sums to unity ($\sum_{i,j} C_{i,j} \approx 1.0$) [2].

Finally, the continuous density of states (DOS) and normalized luminescence curves $L(\hbar\omega)$ are projected onto a uniform grid with resolution $dE$ by convoluting the discrete line spectrum with a Gaussian profile of width $\sigma = 0.70 \cdot \omega_{2}$ [1]:

$$\text{DOS}(E) = \sum_{i,j} C_{i,j} \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(E_{i,j} - E)^{2}}{2\sigma^{2}} \right)$$

$$L(E) = \text{DOS}(E) \cdot E^{3}$$

---

## 5. Phonon Mode Localization

### Inverse Participation Ratio (IPR)

To identify which phonon modes are spatially localized near the defect, defectpl computes two
complementary IPR metrics for each normal mode $k$.

The per-atom participation weight is (Alkauskas 2014, eq. 13):

$$p_{k;\alpha} = \sum_i \Delta r_{k;\alpha i}^2$$

where $\Delta r_{k;\alpha i}$ is the normalized phonon eigenvector component for atom $\alpha$,
direction $i$, in mode $k$.

#### Traditional (condensed-matter) IPR — `iprs` attribute

$$\mathrm{IPR}_k^{\mathrm{trad}} = \frac{\displaystyle\sum_\alpha p_{k;\alpha}^2}{\left(\displaystyle\sum_\alpha p_{k;\alpha}\right)^2}$$

| Value | Interpretation |
|-------|----------------|
| $1$ | Fully localized on one atom |
| $1/N$ | Fully delocalized over $N$ atoms |

Large = more localized.  Scale-invariant: gives identical results for normalized and un-normalized
eigenvectors.

#### Alkauskas-convention IPR — `iprs_alkauskas` attribute

Defined by Alkauskas *et al.* (2014), eq. 12:

$$\mathrm{IPR}_k = \frac{\left(\displaystyle\sum_\alpha p_{k;\alpha}\right)^2}{\displaystyle\sum_\alpha p_{k;\alpha}^2}$$

For phonopy-normalized eigenvectors this simplifies to $1/\sum_\alpha p_{k;\alpha}^2$.

| Value | Interpretation |
|-------|----------------|
| $1$ | Fully localized on one atom |
| $N$ | Fully delocalized over $N$ atoms |

Small = more localized.  This is the reciprocal of the traditional IPR.

#### Relationship between the two conventions

$$\mathrm{IPR}_k^{\mathrm{Alkauskas}} = \frac{1}{\mathrm{IPR}_k^{\mathrm{trad}}}$$

### Localization ratio — `localization_ratio` attribute

Alkauskas *et al.* (2014), eq. 14, define the **localization ratio** $\beta_k$ as:

$$\beta_k = \frac{M}{\mathrm{IPR}_k^{\mathrm{Alkauskas}}} = M \cdot \mathrm{IPR}_k^{\mathrm{trad}}$$

where $M$ is the total number of atoms in the supercell.  In defectpl:

```python
self.localization_ratio = self.natoms * self.iprs   # = natoms × IPR_trad = β_k
```

| $\beta_k$ | Interpretation |
|-----------|----------------|
| $\approx 1$ | Delocalized bulk phonon |
| $\approx M$ | Fully localized on a single atom |

A large $\beta_k$ combined with a large $S_k$ identifies a phonon resonance that couples strongly
to the optical transition *and* is spatially confined near the defect — the diagnostic signature
of a localized vibrational mode.

---

## 6. Finite-Temperature Extension

At temperatures above 0 K, phonons are thermally occupied and both emission and absorption of
phonons contribute to the optical lineshape.  The formalism follows Jin *et al.* (2021) [3].

### 6.1 Bose-Einstein Phonon Occupation

The mean phonon occupation of mode $k$ at temperature $T$ is:

$$\bar{n}_k(T) = \frac{1}{\exp\!\left(\dfrac{\hbar\omega_k}{k_B T}\right) - 1}$$

At $T = 0$, $\bar{n}_k = 0$ and the T=0 formalism is recovered exactly.

### 6.2 Thermal Spectral Density

The thermal extension adds a new spectral density $C(\hbar\omega, T)$ defined as:

$$C(\hbar\omega, T) = \sum_k \bar{n}_k(T)\, S_k\, \delta(\hbar\omega - \hbar\omega_k)$$

This quantity encodes the contribution of thermally occupied phonons to the lineshape.
The time-domain counterpart is:

$$C(t, T) = \int_0^\infty C(\hbar\omega, T)\, e^{i\omega t}\, d(\hbar\omega)$$

computed numerically as $C(t) = \mathcal{F}^{-1}[C(\omega)]$ via FFT (identical to how $S(t)$
is obtained from $S(\omega)$).

### 6.3 Temperature-Dependent Generating Function

The generating function generalises to [3, Eq. 7]:

$$G(t, T) = \exp\!\Big[S(t) - S(0) + 2C(t,T) - 2C(0,T)\Big]\, e^{-\gamma|t|}$$

where:
- $S(0) = \sum_k S_k$ is the total Huang-Rhys factor $S_{\mathrm{HR}}$,
- $C(0, T) = \sum_k \bar{n}_k(T)\, S_k$ is the thermal sum.

At $T = 0$, $C(t,T) = C(0,T) = 0$ and the original formula is recovered.

### 6.4 Absorption Spectrum

The absorption spectral function is obtained from the complex conjugate of the PL generating
function [3, Eq. 8]:

$$G_{\mathrm{abs}}(t, T) = G^*(t, T)$$

because replacing $S(t)$ with its complex conjugate $S^*(-t)$ is equivalent to reversing
the phonon-energy axis: phonon sidebands appear on the **high-energy side** of the ZPL
(phonon absorption raises the photon energy), opposite to the PL emission sideband.

The absorption intensity uses a linear $\omega$ prefactor (photon density of states in
absorption) rather than the $\omega^3$ factor used for emission:

$$\alpha(\hbar\omega) \propto \omega\, A_{\mathrm{abs}}(\hbar\omega)$$

### 6.5 Frequency-Dependent Gaussian Broadening

When `sigma` is supplied as a 2-tuple `(σ_low, σ_high)`, the broadening width varies linearly
with phonon frequency:

$$\sigma(\omega_k) = \sigma_{\mathrm{low}} +
\frac{(\sigma_{\mathrm{high}} - \sigma_{\mathrm{low}})(\omega_k - \omega_{\min})}
{\omega_{\max} - \omega_{\min}}$$

A scalar `sigma` value is equivalent to setting $\sigma_{\mathrm{low}} = \sigma_{\mathrm{high}}$.

### 6.6 Effective Phonon Frequency

A single characteristic phonon frequency that captures the mass-weighted average coupling is [3, Eq. 16]:

$$\Omega_{\mathrm{eff}} = \sqrt{\frac{\sum_k \omega_k^2\, \Delta Q_k^2}{\sum_k \Delta Q_k^2}}$$

---

## 7. Software Pipeline Mapping

The math detailed in this tutorial maps directly to modules in the `defectpl` repository:
* **`defectpl.defectpl.Photoluminescence`**: Rehydrates the cached JSON properties (frequencies $\omega_{k}$, partial factors $S_{k}$, force matrices, and energy grids), manages the evaluation of equations for $S(\hbar\omega)$, and executes the numeric integration of $G(t)$ to transform it into the final luminescence vector $L(\hbar\omega)$.
* **`defectpl.defectpl.VibrationalSpectra1D`**: Handles standalone effective coordinate lineshape synthesis using parameterized 1D potentials.
* **`defectpl.utils.extract_important_properties`**: Formats and extracts the single-value scalar properties ($E_{\text{ZPL}}$, $S$, $w_{\text{ZPL}}$, $\Delta Q$, $\Delta R$) and labels the run mode based on whether displacement or force-based tracking was invoked.

---

## References
[1] Alkauskas, A., Buckley, B. B., Awschalom, D. D., & Van de Walle, C. G. (2014). First-principles theory of the luminescence lineshape for the triplet transition in diamond NV centres. *New Journal of Physics*, 16(7), 073026. https://doi.org/10.1088/1367-2630/16/7/073026

[2] Alkauskas, A., Yan, Q., & Van de Walle, C. G. (2014). First-principles theory of nonradiative carrier capture via multiphonon emission. *Physical Review B*, 90(7), 075202. https://doi.org/10.1103/PhysRevB.90.075202

[3] Jin, Y., Govoni, M., Wolfowicz, G., Rice, S. E., Heremans, F. J., Awschalom, D. D., & Galli, G. (2021). Photoluminescence spectra of point defects in semiconductors: Validation of first-principles calculations. *Physical Review Materials*, 5(8), 084603. https://doi.org/10.1103/PhysRevMaterials.5.084603