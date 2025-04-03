#!/bin/bash 
#SBATCH --time=00:05:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m4818
#SBATCH --mail-user=ayz2@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH -G 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J climate_large
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"

config_file=./configs/large_perl.yaml

module load conda
conda activate /pscratch/sd/a/ayz2/envs/climate

# OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export NCCL_IB_DISABLE=1

srun -n 4 -c 32 -G 4 python train.py --config=$config_file