DefectPL
========

A high-performance computational toolkit for calculating and visualizing
the photoluminescence (PL) spectra, electron-phonon coupling
characteristics, and optical lineshapes of quantum defect centers in
insulators and semiconductors.

|PyPI Version| |Conda Recipe| |Anaconda Version| |Downloads| |License:
MIT|

   ⚠️ **Development Status:** This package is under active development.
   Features, APIs, and documentation are subject to rapid updates.

--------------

📌 Overview & Key Features
--------------------------

**DefectPL** implements photoluminescence calculation framework based on
standard generating function methodologies (*New J. Phys.* **16** 073026
(2014)) to evaluate electronic-vibrational coupling profiles of point
defects in solids from first-principles data. It bridges ab initio
electronic structure outputs with experimentally observable optical
lineshapes, specifically supporting high Huang-Rhys (HR) factor regimes.

The core engine provides automated pipelines to compute, serialize, and
visualize: \* **Macroscopic Optical Lineshapes:** Full photoluminescence
(PL) spectra sidebands accounting for multi-phonon convolutions. \*
**Coupling Parameters:** Quantified total and mode-resolved partial
Huang-Rhys factors (:math:`S_k`), alongside temperature-dependent
Debye-Waller factors (:math:`I_{\text{ZPL}}/I_{\text{tot}}`). \*
**Phonon Localization Metrics:** Spatial confinement analytics via
Inverse Participation Ratios (IPR) and structural localization index
mappings. \* **Spectral Density Mapping:** Multi-mode Electron-Phonon
Spectral Density functions, :math:`S(\omega)`. \* **Isotope
Engineering:** Analytical evaluation of localized isotope substitution
impacts on vibrational mode coupling pathways.

--------------

📚 Documentation & Reference
----------------------------

For comprehensive API references, mathematical formulations, and
step-by-step tutorials, visit the official documentation portal: 👉
https://Shibu778.github.io/defectpl/

Citation
~~~~~~~~

If you utilize DefectPL in your peer-reviewed scientific workflows,
please cite the following original research works:

   📄 **Carbon with Stone-Wales Defect as Quantum Emitter in h-BN**,
   *Phys. Rev. B* **111**, 104109 (2025). `DOI:
   10.1103/PhysRevB.111.104109 <https://doi.org/10.1103/PhysRevB.111.104109>`__

..

   📄 **High-throughput Computational Search for Group-IV-related
   Quantum Defects as Spin-photon Interfaces in 4H-SiC**, *Phys. Rev. B*
   **112**, 184112 (2025). `DOI:
   10.1103/PhysRevB.112.184112 <https://doi.org/10.1103/PhysRevB.112.184112>`__

--------------

🚀 Installation
---------------

DefectPL can be seamlessly integrated via your preferred package manager
ecosystem.

Standard Installation via PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install defectpl

Stable Pre-compiled Binaries via Conda-Forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   conda install conda-forge::defectpl

Editable Source Build (For Developers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone [https://github.com/Shibu778/defectpl.git](https://github.com/Shibu778/defectpl.git)
   cd defectpl
   pip install -e .

--------------

🧑‍💻 Architectural Tracks & Examples
----------------------------------

DefectPL natively exposes **two core calculation modalities**:
**Displacement Mode** (evaluating structural coordinate shift vectors)
and **Force Mode** (evaluating vertical electronic excitation forces).
All core engine classes inherit from Monty’s ``MSONable``, ensuring
atomic state serialization into lightweight JSON formats.

1. Displacement Mode (Structure Coordinates Tracking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ideal when relaxed atomic geometries for both the ground state (GS) and
excited state (ES) are completely resolved alongside Phonopy supercell
calculations.

.. code:: python

   from pathlib import Path
   from pymatgen.core import Structure
   from monty.serialization import dumpfn

   from defectpl.phonon import read_band_yaml
   from defectpl.io.vasp import calc_dR
   from defectpl.defectpl import Photoluminescence

   # 1. Parse ground/excited state geometry and Phonopy coordinates
   struct_gs = Structure.from_file("CONTCAR_GS")
   struct_es = Structure.from_file("CONTCAR_ES")
   frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

   # 2. Extract periodic-boundary-condition safe displacement matrices
   dR = calc_dR(struct_gs, struct_es)

   # 3. Initialize core execution engine
   pl_engine = Photoluminescence(
       frequencies=frequencies,
       eigenvectors=eigenvectors,
       masses=masses,
       dR=dR,          # Pass dR matrix for Displacement mode
       dF=None,
       EZPL=1.95,
       gamma=2.0
   )

   # 4. Generate publication-quality graphics & serialize configuration states
   pl_engine.generate_plots(out_dir="./plots", fig_format="png")
   dumpfn(pl_engine, "properties.json", indent=4)

2. Force Mode (Force Difference Matrix Tracking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ideal for high-throughput pipelines, utilizing the force landscape
acting on the ground-state structure under a vertical electronic
excitation constraint.

.. code:: python

   from defectpl.io.vasp import prepare_dF_files
   from defectpl.defectpl import Photoluminescence
   from defectpl.phonon import read_band_yaml

   # 1. Parse standard Phonopy baseline calculations
   frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

   # 2. Extract force difference vectors (dF = F_excited - F_ground) from VASP output streams
   dF = prepare_dF_files("OUTCAR_GS", "OUTCAR_ES")

   # 3. Execute solver via Force Matrix track
   pl_engine = Photoluminescence(
       frequencies=frequencies,
       eigenvectors=eigenvectors,
       masses=masses,
       dR=None,
       dF=dF,          # Pass dF matrix for Force mode
       EZPL=1.95,
       gamma=2.0
   )
   pl_engine.generate_plots(out_dir="./plots", fig_format="png")

--------------

🤝 Contributing & Bug Reporting
-------------------------------

We welcome community contributions, optimization proposals, and workflow
suggestions! If you uncover numerical bugs or wish to request feature
additions, please systematically log them through our official `GitHub
Issues <https://www.google.com/search?q=https://github.com/Shibu778/defectpl/issues>`__
portal or submit a structured Pull Request.

--------------

👤 Maintainers & Acknowledgements
---------------------------------

**Project Lead Maintainers:** \* **Shibu Meher** \* **Manoj Dey**

Special Acknowledgements
~~~~~~~~~~~~~~~~~~~~~~~~

The development of this software was supported and inspired by several
foundational open-source packages within the materials physics
community: \* **``PyPhotonics``** \* **``nonrad``** \* **``sumo``** \*
**``phonopy``**

.. |PyPI Version| image:: https://img.shields.io/pypi/v/defectpl.svg?color=blue
   :target: https://pypi.org/pypi/defectpl
.. |Conda Recipe| image:: https://img.shields.io/badge/recipe-defectpl-green.svg
   :target: https://github.com/conda-forge/defectpl-feedstock
.. |Anaconda Version| image:: https://anaconda.org/conda-forge/defectpl/badges/version.svg
   :target: https://anaconda.org/conda-forge/defectpl
.. |Downloads| image:: https://static.pepy.tech/badge/defectpl
   :target: https://pepy.tech/project/defectpl
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
