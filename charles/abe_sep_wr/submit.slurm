#!/bin/bash
#SBATCH -A CFD222
#SBATCH -J abe_sep_wr
#SBATCH -o log-%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=mpw2@stanford.edu
#SBATCH -t 2:00:00
#SBATCH -p batch
# #SBATCH -q debug

#SBATCH -N 10
#SBATCH -n 80
#SBATCH --ntasks-per-node 8
#SBATCH -c 1
#SBATCH --threads-per-core=1
#SBATCH -m block:cyclic
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

date

srun -N10 -n80 -c1 --ntasks-per-node 8 --cpu-bind=threads --threads-per-core=1 -m block:cyclic --gpus-per-task=1 --gpu-bind=closest /ccs/proj/$PROJ_CODE/mpw2/cascade/usr_blprobe/cascade/bin/charles_ig_gpu.exe -i charles.in

date

