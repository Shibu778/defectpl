#!/bin/bash

# Script to get band.yaml from vasprun.xml file of DFPT phonon calculation in VASP
# Activate the required conda environment: conda activate defectpl_dev

# Generate the FORCE_CONSTANTS file from vasprun.xml
defectpl phonon-fc vasprun.xml.gz

gunzip POSCAR.gz

defectpl phonon-band --poscar POSCAR --fc FORCE_CONSTANTS --dim "1 1 1"

gzip band.yaml
gzip POSCAR
gzip FORCE_CONSTANTS

