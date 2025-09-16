#!/bin/bash

#SBATCH --job-name=Cross_Set2_SurgicalSAM_Test
# SBATCH --job-name=EN17_Set2_SurgicalSAM_Test
#SBATCH --partition=gpu-v100s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=Logs/thres0.5_test_endovis17_18_%j_%x.out
#SBATCH --error=Logs/thres0.5_test_endovis17_18_%j_%x.err
# SBATCH --output=Logs/finetune_ori_label_test_endovis17_18_%j_%x.out
# SBATCH --error=Logs/finetune_ori_label_test_endovis17_18_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --qos=long
#SBATCH --time=5:00:00

# module load miniconda
conda activate surgicalsam

# python inference.py  --dataset endovis_2018
# python inference.py  --dataset endovis_2017  --fold 0
# python inference.py  --dataset endovis_2017  --fold 1
# python inference.py  --dataset endovis_2017  --fold 2
# python inference.py  --dataset endovis_2017  --fold 3


echo "=== CROSS DATASET INFERENCE (SET 2 EN17 -> EN18) ==="
echo "=== VALIDATION ==="
echo "=== fold 0 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "val"
echo "=== fold 1 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 1 --test_dataset_type "val"
echo "=== fold 2 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 2 --test_dataset_type "val"
echo "=== fold 3 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 3 --test_dataset_type "val"

echo "=== TRAINING ==="
echo "=== fold 0 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 0 --test_dataset_type "train" 
echo "=== fold 1 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 1 --test_dataset_type "train" 
echo "=== fold 2 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 2 --test_dataset_type "train"
echo "=== fold 3 ==="
python cross_inference.py --train_dataset endovis_2017 --test_dataset endovis_2018 --dataset_version set2 --fold 3 --test_dataset_type "train"


# echo "=== CROSS DATASET INFERENCE (SET 3 EN18 -> EN17) ==="
# echo "=== VALIDATION ==="
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 0 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 1 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 2 --test_dataset_type "val"
# python cross_inference.py --train_dataset endovis_2018 --test_dataset endovis_2017 --dataset_version set3 --fold 3 --test_dataset_type "val"
