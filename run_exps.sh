#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=10G
#SBATCH --nodelist=login01
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate ncollapse
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -m scripts.python.$1 1e-0 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -m scripts.python.$1 1e-1 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -m scripts.python.$1 1e-2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -m scripts.python.$1 1e-3 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -m scripts.python.$1 1e-4 &
wait