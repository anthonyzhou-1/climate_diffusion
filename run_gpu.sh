#!/bin/bash 
#SBATCH --time=00:05:00
#SBATCH -C gpu
#SBATCH --account=m4818
#SBATCH --mail-user=ayz2@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J climate_small

config_file=./configs/small_perl.yaml

module load conda
conda activate /pscratch/sd/a/ayz2/envs/climate

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python train.py --config=$config_file