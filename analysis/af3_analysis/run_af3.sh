#!/bin/bash
#SBATCH -J run_af3                     # Job name
#SBATCH -o run_af3_%j.out              # Output file (%j expands to jobID)
#SBATCH -e run_af3_%j.err              # Error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=jc3668@cornell.edu      # Email address to send results to.
#SBATCH -N 1                                # Total number of nodes requested
#SBATCH -n 4                                # Total number of cores requested
#SBATCH --get-user-env                      # Retrieve the user's login environment
#SBATCH --partition=gpu                     # Request partition (change as needed)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --mem=16G                           # Request 16GB of memory
#SBATCH -t 24:00:00                         # Set a time limit of 24 hours

source ~/miniconda3/bin/activate

conda activate alphafold

cd ~/ppi-evaluation/analysis/af3_analysis

python3 -u run_af3.py

