#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n ncollapse python=3.10
  conda activate ncollapse
  mkdir pip-build

  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia --yes
  conda install -c conda-forge scikit-learn seaborn --yes
  conda install -c conda-forge clearml wandb tensorboard --yes
  conda install -c conda-forge tqdm omegaconf --yes
  conda env export | grep -v "^prefix: " > environment.yml
fi
