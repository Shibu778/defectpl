from defectpl.defectpl import DefectPl


def comparepl(
    band_yaml_files,
    contcar_gs,
    contcar_es,
    out_dir,
    EZPL,
    gamma,
    plot_all,
    iplot_xlim,
    file_name="intensity_vs_penergy.pdf",
    plot=False,
):
    all_defctpl = []
    for band_yaml in band_yaml_files:
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
        all_defctpl.append(defctpl)
    xlim = iplot_xlim
    out_path = Path(out_dir) / file_name
    plt.figure(figsize=(4, 4))
    for defectpl in all_defectpl:
        plt.plot(defectpl.I.__abs__(), "k")
    plt.ylabel(r"$I(\hbar\omega)$")
    plt.xlabel(r"Photon energy (eV)")
    plt.xlim(xlim[0], xlim[1])
    x_values, labels = plt.xticks()
    labels = [float(x) / all_defectpl[0].resolution for x in x_values]
    plt.xticks(x_values, labels)
    # plt.ylim(0, 1500)
    if plot:
        plt.show()
    else:
        form = file_name.split(".")[-1]
        if form:
            form = form.lower()
        else:
            form = "pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", format=form)
        plt.close()
