#!/bin/bash
#SBATCH -A CFD209
#SBATCH -J m16_wr
#SBATCH -o log-%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=mpw2@stanford.edu
#SBATCH -t 2:00:00
#SBATCH -p batch
# #SBATCH -q debug

#SBATCH -N 5
#SBATCH -n 40
#SBATCH --ntasks-per-node 8
#SBATCH -c 1
#SBATCH --threads-per-core=1
#SBATCH -m block:cyclic
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

date

srun -N5 -n40 -c1 --ntasks-per-node 8 --cpu-bind=threads --threads-per-core=1 -m block:cyclic --gpus-per-task=1 --gpu-bind=closest /ccs/proj/$PROJ_CODE/mpw2/cascade/usr_blprobe/cascade/bin/charles_ig_gpu.exe -i charles.in

date

