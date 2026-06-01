#!/bin/bash

from=/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/raw_data/dfpt_phonon

cp $from/band.yaml .
cp $from/irreps.yaml .
cp $from/vasprun.xml .
cp $from/POSCAR .
cp $from/OUTCAR .

gzip *