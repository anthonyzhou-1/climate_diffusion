#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J process_data
#SBATCH --mail-user=ayz2@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 2:00:00

module load conda
conda activate /pscratch/sd/a/ayz2/envs/climate

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 256 --cpu_bind=cores python process_data.py