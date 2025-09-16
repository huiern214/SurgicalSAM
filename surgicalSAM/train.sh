#!/bin/bash

#SBATCH --job-name=fold0_aug_finetune_en17_18_train_endovis
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=Logs/finetune_en17set2model_train_endovis18_%j_%x.out
#SBATCH --error=Logs/finetune_en17set2model_train_endovis18_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --qos=long
#SBATCH --time=12:00:00

# module load miniconda
conda activate surgicalsam

# python train.py  --dataset endovis_2018
python train_finetune.py  --dataset endovis_2018  --fold 0
