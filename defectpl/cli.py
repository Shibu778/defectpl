import click


@click.command()
@click.option(
    "--band_yaml",
    default="./band.yaml",
    help="File path absolute or relative including the filename",
)
@click.option(
    "--contcar_gs",
    default="./CONTCAR_gs",
    help="File path absolute or relative including the filename",
)
@click.option(
    "--contcar_es",
    default="./CONTCAR_es",
    help="File path absolute or relative including the filename",
)
@click.option(
    "--out_dir",
    default="./",
    help="File path absolute or relative including the filename",
)
@click.option(
    "--ezpl",
    default=1.95,
    help="Energy of the zero-phonon line in eV",
)
@click.option(
    "--gamma",
    default=2,
    help="Gamma value for the phonon broadening",
)
@click.option(
    "--plot_all",
    is_flag=True,
    default=False,
    help="Plot all available data",
)
@click.option(
    "--iplot_xlim",
    default=[1000, 2000],
    type=(int, int),
    help="X-axis limits (in meV) for the intensity vs photon energy plot",
)
@click.option(
    "--fig_format",
    default="svg",
    help="Figure format for saving plots (e.g., svg, png, pdf)",
)
def main(
    band_yaml,
    contcar_gs,
    contcar_es,
    out_dir,
    ezpl,
    gamma,
    plot_all,
    iplot_xlim,
    fig_format,
):
    """Main function to run the DefectPL analysis and plotting."""
    from defectpl.defectpl import DefectPl
    from defectpl.plot import Plotter

    dpl = DefectPl(
        band_yaml,
        contcar_gs,
        contcar_es,
        ezpl,
        gamma,
        iplot_xlim=iplot_xlim,
        plot_all=plot_all,
        out_dir=out_dir,
        fig_format=fig_format,
    )
