#!/bin/bash
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --qos=normal
#SBATCH --job-name=CA1
#SBATCH --output=runCA1.out
#SBATCH --time 0-02:00

rm -rf output

echo "Running CA1 model at $(date)"

mpiexec nrniv -mpi -quiet -python run_network.py simulation_config.json

echo "Done running CA1 model at $(date)"
