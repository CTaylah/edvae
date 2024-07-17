#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --time=24:00:00 ## time it will take to complete job

#SBATCH --partition=gpu ##Partition I want to use

#SBATCH --gpus-per-node=1

#SBATCH --cpus-per-task=2

#SBATCH --mem-per-cpu=82G

#SBATCH --job-name=nn_training ## Name of job

#SBATCH --output=vae_gan.%j.out ##Name of output file

module load py-venv-ml/nightly
export PYTHONPATH=$PYTHONPATH:/mnt/home/taylcard/dev/EDVAE/
python main.py 

