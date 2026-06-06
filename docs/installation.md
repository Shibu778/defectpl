# Installation Guide

This page covers the different methods available to install **DefectPL**, depending on your workflow requirements.

---

## 📋 Prerequisites

Before installing, ensure your environment meets the following requirements:
- **Python:** version 3.9 or higher (fully optimized for Python 3.12+)
- **Core Dependencies:** `numpy`, `scipy`, `matplotlib`, `pymatgen`, `monty`, and `pyyaml` (these will be automatically configured during installation)

---

## 🚀 Installation Options

### Option 1: Via PyPI (Recommended for standard users)
The easiest way to install the stable release of DefectPL is directly from the Python Package Index using `pip`:

```bash
pip install defectpl

```

### Option 2: Via Conda / Anaconda

If you use Anaconda or a localized conda environment architecture, you can pull the pre-compiled package from the `conda-forge` channel:

```bash
conda install conda-forge::defectpl

```

### Option 3: From GitHub Source (Recommended for developers)

If you want to contribute to the package, inspect the code, or work with the latest development branches, clone the repository directly and execute an editable installation:

```bash
# Clone the repository
git clone [https://github.com/Shibu778/defectpl.git](https://github.com/Shibu778/defectpl.git)

# Move into the root directory tracking setup files
cd defectpl

# Install in editable/development mode
pip install -e .

```

---

## Verification

To verify that the installation was successful and the package pathways are accessible globally, open a terminal window or environment shell and run:

```bash
python -c "import defectpl; print(defectpl.__version__)"

```

If it prints the package version number without throwing an `ImportError`, your development suite is completely ready to run photoluminescence calculations.