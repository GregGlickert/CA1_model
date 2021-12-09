#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos=normal
#SBATCH --job-name=CA1
#SBATCH --output=buildCA1.out
#SBATCH --time 0-01:00

echo "building CA1 model at $(date)"

rm -rf network

python3 build_network.py

echo "Done building CA1 at $(date)"
