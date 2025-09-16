#!/bin/bash

# SBATCH --job-name=thres0.5_fold3_SurgicalSAM_Preprocess_2018
#SBATCH --job-name=EN18_Preprocess_sept1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=Logs/preprocess_%j_%x.out
#SBATCH --error=Logs/preprocess_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --qos=long
#SBATCH --time=48:00:00

conda activate surgicalsam
# conda activate /home/user/huiern214/miniconda3/envs/surgicalsam
cd surgicalSAM/tools/
# python data_preprocess.py  --dataset endovis_2018  --n-version 40
# python data_preprocess.py  --dataset endovis_2018  --n-version 40 --fold 0
# python data_preprocess.py  --dataset endovis_2018  --n-version 40 --fold 1
# python data_preprocess.py  --dataset endovis_2018  --n-version 40 --fold 2
# python data_preprocess.py  --dataset endovis_2018  --n-version 40 --fold 3
python data_preprocess.py  --dataset endovis_2018  --n-version 40