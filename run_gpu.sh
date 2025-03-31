#!/bin/bash 
#SBATCH --time=24:00:00
#SBATCH -C gpu
#SBATCH --account=m4818
#SBATCH --mail-user=ayz2@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J clima_dit

config_file=./configs/clima_dit_sphere.yml

module load conda
conda activate /pscratch/sd/a/ayz2/envs/climate

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 torchrun --standalone --nnodes=1 --nproc-per-node=1 train_weather_refiner.py --config=$config_file