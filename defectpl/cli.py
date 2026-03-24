import json
from pathlib import Path

import click


@click.group()
def main():
    """defectpl command group."""
    pass


@main.command(name="run")
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
def run(
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
    """Main function to run the DefectPl analysis and plotting."""
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


@main.command(name="dq")
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    default=None,
    help="Optional output file to write JSON with result.",
)
@click.option(
    "--format",
    "-f",
    "out_format",
    type=click.Choice(["plain", "json"], case_sensitive=False),
    default="plain",
    help="Print format (plain = number, json = JSON object).",
)
def dq(structure1, structure2, out_path, out_format):
    """Calculate deltaQ between two structure files.

    Usage:
      defectpl dq CONTCAR_gs CONTCAR_es [--out result.json] [--format json]
    """
    from defectpl.utils import calc_deltaQ

    try:
        delta_q = calc_deltaQ(structure1, structure2)
    except Exception as exc:
        raise click.ClickException(f"Failed to calculate deltaQ: {exc}")

    if out_format.lower() == "plain":
        click.echo(f"{delta_q:.8f}")
    else:
        click.echo(
            json.dumps(
                {
                    "structure1": str(structure1),
                    "structure2": str(structure2),
                    "deltaQ": delta_q,
                },
                indent=2,
            )
        )

    if out_path:
        p = Path(out_path)
        p.write_text(
            json.dumps(
                {
                    "structure1": str(structure1),
                    "structure2": str(structure2),
                    "deltaQ": delta_q,
                },
                indent=2,
            )
        )
        click.echo(f"Wrote result to {p}")


if __name__ == "__main__":
    main()
