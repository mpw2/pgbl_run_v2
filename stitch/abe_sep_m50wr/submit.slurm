#!/bin/bash
#SBATCH -A CFD222
#SBATCH -J stitch_abe
#SBATCH -o log-%x-%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mpw2@stanford.edu
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 10

date

srun -N 10 -n 560 --ntasks-per-node 56 /ccs/proj/$PROJ_CODE/mpw2/cascade/usr_dev/cascade/bin/stitch.exe

date

