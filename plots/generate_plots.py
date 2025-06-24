# Generate Plots for the documentation
# Example Usage of the DefectPL library

from defectpl.defectpl import DefectPl
from defectpl.plot import Plotter
from defectpl.vibspectra1d import VibrationalSpectra1D

band_yaml = "../../tests/data/band.yaml"
contcar_gs = "../../tests/data/CONTCAR_gs"
contcar_es = "../../tests/data/CONTCAR_es"
out_dir = "."
EZPL = 1.95
gamma = 2
plot_all = False
iplot_xlim = [1000, 2000]
fig_format = "svg"

dpl = DefectPl(
    band_yaml,
    contcar_gs,
    contcar_es,
    EZPL,
    gamma,
    iplot_xlim=iplot_xlim,
    plot_all=plot_all,
    out_dir=out_dir,
    fig_format=fig_format,
)

plotter = Plotter()
max_freq = None  # Max frequence to plot S(omega)
iylim = None  # Intensity y-limit for the intensity vs photon energy plot
iplot_xlim = iplot_xlim  # x-limit for the intensity vs photon
plotter.plot_penergy_vs_pmode(
    frequencies=dpl.frequencies,
    plot=False,
    out_dir=out_dir,
    fig_format=fig_format,
)
# Plot IPR vs phonon energy
plotter.plot_ipr_vs_penergy(
    dpl.frequencies,
    dpl.iprs,
    plot=False,
    out_dir=out_dir,
    fig_format=fig_format,
)
# Plot localization ratio vs phonon energy
plotter.plot_loc_rat_vs_penergy(
    dpl.frequencies,
    dpl.localization_ratio,
    plot=False,
    out_dir=out_dir,
    fig_format=fig_format,
)

# Plot vibrational displacement vs phonon energy
plotter.plot_qk_vs_penergy(
    dpl.frequencies,
    dpl.qks,
    plot=False,
    out_dir=out_dir,
    fig_format=fig_format,
)
# Plot partial HR factor vs phonon energy
plotter.plot_HR_factor_vs_penergy(
    dpl.frequencies,
    dpl.Sks,
    plot=False,
    out_dir=out_dir,
    fig_format=fig_format,
)
# Plot S(omega) vs phonon energy
plotter.plot_S_omega_vs_penergy(
    dpl.frequencies,
    dpl.S_omega,
    dpl.omega_range,
    plot=False,
    out_dir=out_dir,
    max_freq=max_freq,
    fig_format=fig_format,
    figsize=(4, 4),
)
# Plot S(omega) and Sks vs phonon energy
plotter.plot_S_omega_Sks_vs_penergy(
    dpl.frequencies,
    dpl.S_omega,
    dpl.omega_range,
    dpl.Sks,
    plot=False,
    out_dir=out_dir,
    max_freq=max_freq,
    fig_format=fig_format,
    figsize=(4, 4),
)
# Plot S(omega) and Sks vs phonon energy
plotter.plot_S_omega_Sks_Loc_rat_vs_penergy(
    dpl.frequencies,
    dpl.S_omega,
    dpl.omega_range,
    dpl.Sks,
    dpl.localization_ratio,
    plot=False,
    out_dir=out_dir,
    max_freq=max_freq,
    fig_format=fig_format,
    figsize=(4, 4),
)
# Plot S(omega), Sks and IPR vs phonon energy
plotter.plot_S_omega_Sks_ipr_vs_penergy(
    dpl.frequencies,
    dpl.S_omega,
    dpl.omega_range,
    dpl.Sks,
    dpl.iprs,
    plot=False,
    out_dir=out_dir,
    max_freq=max_freq,
    fig_format=fig_format,
    figsize=(4, 4),
)
# Plot intensity vs photon energy
plotter.plot_intensity_vs_penergy(
    dpl.frequencies,
    dpl.I,
    dpl.resolution,
    iplot_xlim,
    plot=False,
    out_dir=out_dir,
    iylim=iylim,
    fig_format=fig_format,
)

# Plotting one-dimensional vibrational spectra
ligo = VibrationalSpectra1D(2.6, 35.75, 41.56, 1.5476, 300, 1.2, 0.001, 1800)
ligo.compute_lineshape()
ligo.plot_lineshape(save_file="./one_d_lineshape.svg", figsize=(4, 4))
