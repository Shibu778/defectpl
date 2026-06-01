#!/bin/bash

# This script is for organizing the data for the PBE level calculation
# System: NV-1 center in diamond

# NOTE: We have observed CDFT calculation in ground state removes the fractional 
# occupancies in the EIGENVAL file. Althought fractional occupancies are not
# observed for this defect. Here, we have tested the CDFT for ground state to
# test the energies and forces, which vary slightly from that without CDFT.

from=/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/raw_data/cdft_gs_es
mkdir -p gs abs ems zpl/frac_occ
cd gs
# Ground state with CDFT
cp -r $from/gs/hse06/{POSCAR,CONTCAR,OUTCAR,OSZICAR,EIGENVAL,INCAR,vasprun.xml,PROCAR} .

cd ../abs
cp -r $from/abs/hse06/{POSCAR,CONTCAR,OUTCAR,OSZICAR,EIGENVAL,INCAR,vasprun.xml,PROCAR} .

cd ../ems
cp -r $from/ems/hse06/{POSCAR,CONTCAR,OUTCAR,OSZICAR,EIGENVAL,INCAR,vasprun.xml,PROCAR} .

cd ../zpl
cp -r $from/zpl/hse06/{POSCAR,CONTCAR,OUTCAR,OSZICAR,EIGENVAL,INCAR,vasprun.xml,PROCAR} .

cd frac_occ
cp -r $from/zpl/frac_occ/hse06/{POSCAR,CONTCAR,OUTCAR,OSZICAR,EIGENVAL,INCAR,vasprun.xml,PROCAR} .