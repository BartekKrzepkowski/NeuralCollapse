#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n clp_env python=3.9
  conda activate clp_env
  mkdir pip-build

  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia --yes
  conda install -c conda-forge scikit-learn seaborn --yes
  conda install -c conda-forge clearml wandb tensorboard --yes
  conda install -c conda-forge tqdm --yes
  conda env export | grep -v "^prefix: " > environment.yml
fi
