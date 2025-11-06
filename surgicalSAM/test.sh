#!/bin/bash

#SBATCH --job-name=exp10_NoKLwithAugmentUnionFinal_Cross_Set2_SurgicalSAM_Test
# SBATCH --job-name=exp9_finetuneNoAugmentV2NEW_Cross_Set2_SurgicalSAM_Test
# SBATCH --job-name=EN17_Set2_SurgicalSAM_Test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=Logs/test_endovis17_18_%j_%x.out
#SBATCH --error=Logs/test_endovis17_18_%j_%x.err
# SBATCH --output=Logs/finetune_ori_label_test_endovis17_18_%j_%x.out
# SBATCH --error=Logs/finetune_ori_label_test_endovis17_18_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --time=5:00:00

# module load miniconda
# conda activate surgicalsam
module load miniforge
conda activate seem

# python inference.py  --dataset endovis_2018
# python inference.py  --dataset endovis_2017  --fold 0
# python inference.py  --dataset endovis_2017  --fold 1
# python inference.py  --dataset endovis_2017  --fold 2
# python inference.py  --dataset endovis_2017  --fold 3


echo "=== CROSS DATASET INFERENCE (SET 2 EN17 -> EN18) ==="
echo "=== VALIDATION ==="
# echo "=== fold 0 ==="
# echo "=== Dice + noKL ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp7/endovis_2018/aug_gpu8_001_DiceFG_noKL/0/model_ckp.pth"
# echo "=== Dice + 0.5allKL ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp7/endovis_2018/aug_gpu8_001_DiceFG_05allKL/0/model_ckp.pth"
# echo "=== fgDice + 0.5bgKL ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp7/endovis_2018/aug_gpu8_001_DiceFG_05klBG/0/model_ckp.pth"
# echo "=== fgDice + bgKL ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp7/endovis_2018/aug_gpu8_001_DiceFG_klBG/0/model_ckp.pth"
# python cross_inference_test.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val"
echo "=== fold 0 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" 
# echo "f0 new4"
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp10/endovis_2018/Dice_noKLwithAugUnion/0/new4/model_ckp.pth"
echo "=== fold 1 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 1 --test_dataset_type "val"
echo "=== fold 2 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 2 --test_dataset_type "val"
# echo "=== fold 3 new4==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 3 --test_dataset_type "val" --checkpoint_path "/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/exp10/endovis_2018/Dice_noKLwithAugUnion/3/new4/model_ckp.pth"
echo "=== fold 3 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 3 --test_dataset_type "val"

# echo "=== TRAINING ==="
# echo "=== fold 0 ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "train" 
# echo "=== fold 1 ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 1 --test_dataset_type "train" 
# echo "=== fold 2 ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 2 --test_dataset_type "train"
# echo "=== fold 3 ==="
# python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 3 --test_dataset_type "train"


# echo "=== CROSS DATASET INFERENCE (SET 3 EN18 -> EN17) ==="
# echo "=== VALIDATION ==="
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 0 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 1 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 2 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 3 --test_dataset_type "val"
