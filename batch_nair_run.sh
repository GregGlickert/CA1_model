#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --qos=normal
#SBATCH --job-name=CA1
#SBATCH --output=RunCA1.out
#SBATCH --time 0-02:00

rm -rf output

echo "Running CA1 model at $(date)"

mpiexec nrniv -mpi -quiet -python run_network.py config.json

echo "Done running CA1 model at $(date)"
