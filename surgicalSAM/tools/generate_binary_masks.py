import os
import cv2
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
# input_root_dir = "../data/endovis_2018/train"
# output_root_dir = "../data/endovis_2018/binary_annotations"
# input_root_dir = "/scr/user/huiern214/SurgicalSAM/data/en17to18_thres0.9/endovis_2018/fold3/train/0/annotations"
# output_root_dir = "/scr/user/huiern214/SurgicalSAM/data/en17to18_thres0.9/endovis_2018/fold3/train/0/binary_annotations"

# input_root_dir = "/scr/user/huiern214/SurgicalSAM/data/en17to18_thres0.9/endovis_2018/fold3/val/annotations"
# output_root_dir = "/scr/user/huiern214/SurgicalSAM/data/en17to18_thres0.9/endovis_2018/fold3/val/binary_annotations"

input_root_dir = "/scr/user/huiern214/SurgicalSAM/surgicalSAM/output_masks_endovis17_to_endovis18_set2/thres0.7/fold3/train/annotations"
output_root_dir = "/scr/user/huiern214/SurgicalSAM/surgicalSAM/output_masks_endovis17_to_endovis18_set2/thres0.7/fold3/train/binary_annotations"
# input_root_dir = "/scr/user/huiern214/SurgicalSAM/surgicalSAM/output_masks_endovis17_to_endovis18_set2/thres0.7/fold3/val/annotations"
# output_root_dir = "/scr/user/huiern214/SurgicalSAM/surgicalSAM/output_masks_endovis17_to_endovis18_set2/thres0.7/fold3/val/binary_annotations"

# ---------------------------
# LOOP OVER ALL SEQUENCES
# ---------------------------
for seq_name in os.listdir(input_root_dir):
    seq_path = os.path.join(input_root_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue

    print(f"Processing sequence: {seq_name}")

    # Create corresponding output dir
    output_seq_dir = os.path.join(output_root_dir, seq_name)
    os.makedirs(output_seq_dir, exist_ok=True)

    # ---------------------------
    # LOOP OVER ALL IMAGES IN THIS SEQ
    # ---------------------------
    for fname in os.listdir(seq_path):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(seq_path, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]  # skip background

        print(f"  Found {unique_classes} in {fname}")

        for cls_id in unique_classes:
            binary_mask = np.where(mask == cls_id, 255, 0).astype(np.uint8)
            out_fname = f"{fname.replace('.png', '')}_class{cls_id}.png"
            out_path = os.path.join(output_seq_dir, out_fname)
            cv2.imwrite(out_path, binary_mask)
            print(f"    Saved {out_path}")
