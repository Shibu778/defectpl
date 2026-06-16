# Installation

## Prerequisites

- Python 3.10 or later
- A working VASP installation is **not** required to run DefectPL — only to produce the input files
  it reads.

---

## Install from PyPI

### Core only (no DFT parser)

```bash
pip install defectpl
```

Provides pure-math utilities, the native PROCAR parser, constants, and CLI commands that do not
depend on pymatgen or phonopy.

### With VASP file support

```bash
pip install "defectpl[vasp]"
```

Adds pymatgen for reading POSCAR, CONTCAR, OUTCAR, EIGENVAL, and vasprun.xml files.

### With phonon support

```bash
pip install "defectpl[phonon]"
```

Adds phonopy for computing force constants and phonon band structures.

### Full install (recommended)

```bash
pip install "defectpl[all]"
```

Equivalent to `defectpl[vasp,phonon]`.

---

## Install from conda-forge

```bash
conda install -c conda-forge defectpl
```

The conda package bundles all optional dependencies.

---

## Install from source

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl
pip install -e ".[all]"
```

The `-e` flag installs in editable mode so local changes take effect immediately without
reinstalling.

---

## Verify the installation

```bash
defectpl --help
python -c "import defectpl; print(defectpl.__version__)"
```

---

## Optional extras summary

| Extra | Additional packages | When you need it |
|-------|---------------------|-----------------|
| `vasp` | pymatgen, pymatgen-core | Reading VASP output files |
| `phonon` | phonopy | Phonon force constants, band.yaml |
| `all` | pymatgen, pymatgen-core, phonopy | Full VASP workflow |

---

## Build the documentation locally

```bash
pip install -r docs/requirements.txt
mkdocs serve          # live-preview at http://127.0.0.1:8000
mkdocs build          # static HTML in site/
```
