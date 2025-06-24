DefectPL
=========

A comprehensive toolkit for calculating and visualizing photoluminescence spectra of quantum defects. It also supports the analysis of other optical properties of point defects in insulators and semiconductors.

.. image:: https://img.shields.io/pypi/v/defectpl.svg
   :target: https://pypi.python.org/pypi/defectpl
.. image:: https://static.pepy.tech/badge/defectpl
   :target: https://pepy.tech/project/defectpl
.. image:: https://img.shields.io/badge/recipe-defectpl-green.svg
   :target: https://github.com/conda-forge/defectpl-feedstock
.. image:: https://anaconda.org/conda-forge/defectpl/badges/version.svg
   :target: https://anaconda.org/conda-forge/defectpl
.. image:: https://img.shields.io/conda/vn/conda-forge/defectpl.svg
   :target: https://anaconda.org/conda-forge/defectpl
.. image:: https://img.shields.io/conda/dn/conda-forge/defectpl.svg
   :target: https://anaconda.org/conda-forge/defectpl
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. warning::

   This package is currently under active development.

Purpose
-------

**DefectPL** is designed to compute the photoluminescence intensity of point defects in solids using the methodology described in *New J. Phys. 16 (2014) 073026*. It also provides tools to calculate and plot related quantities such as:

- Partial Huang-Rhys factors
- Huang-Rhys factor
- Debye-Waller factor
- Inverse participation ratios (IPR)
- Localization ratios
- Vibrational displacements
- Effect of Isotope substitution
- Photoluminescence Spectra in the High Huang-Rhys Factor Regime

If you use this package in your research, please consider citing:

- *Carbon with Stone-Wales defect as quantum emitter in h-BN*, Phys. Rev. B 111, 104109 (2025): https://doi.org/10.1103/PhysRevB.111.104109
- *High-throughput computational search for group-IV-related quantum defects as spin-photon interfaces in 4H-SiC*, ChemRxiv (2025): https://doi.org/10.26434/chemrxiv-2025-7whnf9

Documentation
-------------

Full documentation is available at: https://Shibu778.github.io/defectpl/

Installation
------------

Install via **pip**::

   pip install defectpl

Install via **conda**::

   conda install conda-forge::defectpl

Install from **GitHub**::

   git clone https://github.com/Shibu778/defectpl.git
   cd defectpl/defectpl
   pip install -e .

Example Usage
-------------

Here’s a minimal example using data for a negative NV center in diamond::

   from defectpl.defectpl import DefectPl

   band_yaml = "../tests/data/band.yaml"
   contcar_gs = "../tests/data/CONTCAR_gs"
   contcar_es = "../tests/data/CONTCAR_es"
   out_dir = "./plots"
   EZPL = 1.95
   gamma = 2
   plot_all = True
   iplot_xlim = [1000, 2000]

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

Contributing
------------

Contributions, suggestions, and bug reports are welcome!  
If you encounter any issues, please open an issue or submit a pull request.

Author
------

**Main Maintainer:** Shibu Meher, Manoj Dey
