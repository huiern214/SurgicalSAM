#!/bin/bash

#SBATCH --job-name=f0_thres07_Dice_exp10_noKLwithAugUnion_dice_finetune_en17_18_train_endovis
# SBATCH --job-name=bce_fold0_aug_finetune_en17_18_train_endovis
# SBATCH --job-name=f3pseudo_noaug_exp10_finetune_en17_18_train_endovis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=Logs/finetune_en17set2model_train_endovis18_%j_%x.out
#SBATCH --error=Logs/finetune_en17set2model_train_endovis18_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --time=36:00:00

# module load miniconda
# conda activate surgicalsam
module load miniforge
conda activate seem

# python train.py  --dataset endovis_2018
# python train_finetune.py  --dataset endovis_2018  --fold 3
# python teacher_student_train.py  --dataset endovis_2018 --loss_type dice --fold 0 --checkpoint_path /home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/set2_en17_ckp/0/model_ckp.pth
# python teacher_student_train3.py  --dataset endovis_2018 --fold 0 --loss_type dice
python teacher_student_train.py  --dataset endovis_2018 --fold 0
