#!/bin/bash

# This script is used to build this data directory with only useful files

from=/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/raw_data/test_vasp660_gpu/ncdft_calc

to=.

# Delete the already existing *.gz files
# find . -name "*.gz" -exec rm {} \;

# copy and gzip the useful files
mkdir -p abs
cp $from/abs/{POSCAR,OUTCAR} $to/abs/
gzip $to/abs/POSCAR
gzip $to/abs/OUTCAR

mkdir -p abs/hse06
cp $from/abs/hse06/{POSCAR,OUTCAR} $to/abs/hse06/
gzip $to/abs/hse06/POSCAR
gzip $to/abs/hse06/OUTCAR


mkdir -p gs
cp $from/gs/{POSCAR,CONTCAR,OUTCAR} $to/gs/
gzip $to/gs/POSCAR
gzip $to/gs/CONTCAR
gzip $to/gs/OUTCAR

mkdir -p gs/hse06
cp $from/gs/hse06/{POSCAR,CONTCAR,OUTCAR} $to/gs/hse06/
gzip $to/gs/hse06/POSCAR
gzip $to/gs/hse06/CONTCAR
gzip $to/gs/hse06/OUTCAR

mkdir -p dfpt
cp $from/dfpt/{POSCAR,vasprun.xml} $to/dfpt/
gzip $to/dfpt/POSCAR
gzip $to/dfpt/vasprun.xml

mkdir -p dfpt/hse06
cp $from/dfpt/hse06/{POSCAR,vasprun.xml} $to/dfpt/hse06/
gzip $to/dfpt/hse06/POSCAR
gzip $to/dfpt/hse06/vasprun.xml

mkdir -p ems
cp $from/ems/{POSCAR,OUTCAR} $to/ems/
gzip $to/ems/POSCAR
gzip $to/ems/OUTCAR

mkdir -p ems/hse06
cp $from/ems/hse06/{POSCAR,OUTCAR} $to/ems/hse06/
gzip $to/ems/hse06/POSCAR
gzip $to/ems/hse06/OUTCAR

mkdir -p zpl
cp $from/zpl/{POSCAR,CONTCAR,OUTCAR} $to/zpl/
gzip $to/zpl/POSCAR
gzip $to/zpl/CONTCAR
gzip $to/zpl/OUTCAR

mkdir -p zpl/hse06
cp $from/zpl/hse06/{POSCAR,CONTCAR,OUTCAR} $to/zpl/hse06/
gzip $to/zpl/hse06/POSCAR
gzip $to/zpl/hse06/CONTCAR
gzip $to/zpl/hse06/OUTCAR

mkdir -p frac_abs
cp $from/frac_abs/{POSCAR,OUTCAR} $to/frac_abs/
gzip $to/frac_abs/POSCAR
gzip $to/frac_abs/OUTCAR

mkdir -p frac_abs/hse06
cp $from/frac_abs/hse06/{POSCAR,OUTCAR} $to/frac_abs/hse06/
gzip $to/frac_abs/hse06/POSCAR
gzip $to/frac_abs/hse06/OUTCAR

mkdir -p frac_zpl
cp $from/frac_zpl/{POSCAR,CONTCAR,OUTCAR} $to/frac_zpl/
gzip $to/frac_zpl/POSCAR
gzip $to/frac_zpl/CONTCAR
gzip $to/frac_zpl/OUTCAR

mkdir -p frac_zpl/hse06
cp $from/frac_zpl/hse06/{POSCAR,CONTCAR,OUTCAR} $to/frac_zpl/hse06/
gzip $to/frac_zpl/hse06/POSCAR
gzip $to/frac_zpl/hse06/CONTCAR
gzip $to/frac_zpl/hse06/OUTCAR

mkdir -p frac_ems
cp $from/frac_ems/{POSCAR,OUTCAR} $to/frac_ems/
gzip $to/frac_ems/POSCAR
gzip $to/frac_ems/OUTCAR

mkdir -p frac_ems/hse06
cp $from/frac_ems/hse06/{POSCAR,OUTCAR} $to/frac_ems/hse06/
gzip $to/frac_ems/hse06/POSCAR
gzip $to/frac_ems/hse06/OUTCAR

mkdir -p mlip_phonon/mattersim
cp $from/gs/CONTCAR $to/mlip_phonon/mattersim/CONTCAR_gs
gzip $to/mlip_phonon/mattersim/CONTCAR_gs

mkdir -p mlip_phonon/mattersim/hse06
cp $from/gs/hse06/CONTCAR $to/mlip_phonon/mattersim/hse06/CONTCAR_gs
gzip $to/mlip_phonon/mattersim/hse06/CONTCAR_gs

mkdir -p mlip_phonon/mace
cp $from/gs/CONTCAR $to/mlip_phonon/mace/CONTCAR_gs
gzip $to/mlip_phonon/mace/CONTCAR_gs

mkdir -p mlip_phonon/mace/hse06
cp $from/gs/hse06/CONTCAR $to/mlip_phonon/mace/hse06/CONTCAR_gs
gzip $to/mlip_phonon/mace/hse06/CONTCAR_gs