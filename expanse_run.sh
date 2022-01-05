#!/bin/bash

#SBATCH --partition shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --account=umc113
#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --mem=8G
#SBATCH --time 0-48:00

module purge
module load slurm
module load cpu
module load intel
module load intel-mpi
module load ncurses


rm -rf output


echo "Running model at $(date)"

#mpirun nrniv -mpi -quiet -python3 run_network.py simulation_config.json
ibrun nrniv -mpi -python run_network.py simulation_config.json
#python run_network.py

echo "Done running model at $(date)"