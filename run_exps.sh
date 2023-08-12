#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=batch
#SBATCH --nodelist=login01
#SBATCH --time=3-0
#SBATCH --output=slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate $HOME/miniconda3/envs/clpi_env
CUDA_VISIBLE_DEVICES=0 python3 -u $1.py &
CUDA_VISIBLE_DEVICES=0 python3 -u $2.py &
CUDA_VISIBLE_DEVICES=0 python3 -u $3.py &
CUDA_VISIBLE_DEVICES=0 python3 -u $4.py &
wait