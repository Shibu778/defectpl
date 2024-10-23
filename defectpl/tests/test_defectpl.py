from defectpl import __version__
from defectpl.defectpl import DefectPl


def test_version():
    assert __version__ == "0.1.0"


def test_DefectPl():
    band_yaml = "./tests/data/band.yaml"
    contcar_gs = "./tests/data/CONTCAR_gs"
    contcar_es = "./tests/data/CONTCAR_es"
    out_dir = "./examples/plots"
    EZPL = 1.95
    gamma = 2
    plot_all = True
    iplot_xlim = [1000, 2000]

    try:
        defctpl = DefectPl(
            band_yaml,
            contcar_gs,
            contcar_es,
            EZPL,
            gamma,
            iplot_xlim=iplot_xlim,
            plot_all=plot_all,
            out_dir=out_dir,
        )
    except Exception as e:
        assert False, e
