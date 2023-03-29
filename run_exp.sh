#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=batch
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $HOME/anaconda3/envs/fp2

python3 -u run_exp.py