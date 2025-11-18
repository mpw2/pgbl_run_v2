#!/bin/bash
#SBATCH -A CFD209
#SBATCH -J stitch
#SBATCH -o log-%x-%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mpw2@stanford.edu
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

date

srun -N 1 -n 56 --ntasks-per-node 56 /ccs/proj/$PROJ_CODE/mpw2/cascade/usr_dev/cascade/bin/stitch.exe

date

