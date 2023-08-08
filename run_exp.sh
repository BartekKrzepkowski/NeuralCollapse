#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --nodelist=login01
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate $HOME/miniconda3/envs/clpi_env
python3 -u $1.py