import os
import numpy as np
import torch
from PIL import Image
import cv2
import os.path as osp
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F

def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def set_torch_image(transformed_mask):
    input_mask = preprocess(transformed_mask)  # pad to 1024
    return input_mask

def set_mask(mask):   
    """ Transform the mask to the form expected by SAM, the transformed mask will be used to generate class embeddings
        Adapated from set_image in the official code of SAM https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py
    """
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask)
    input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_mask = set_torch_image(input_mask_torch)
    
    return input_mask

def generate_class_embeddings_from_existing_features(
    mask_dir="../../data/endovis_2018/val/binary_annotations",
    feat_dir="../../data/endovis_2018/val/sam_features_h",
    class_embedding_save_dir="../../data/endovis_2018/val/class_embeddings_h",
):
    """
    Go through each frame and mask in the dataset,
    use the precomputed SAM features and masks to compute
    class embeddings (mean feature vectors for masked regions).

    Saves the embeddings to class_embedding_save_dir.

    Args:
        mask_dir: path to binary annotation masks
        feat_dir: path to sam_features_h directory
        class_embedding_save_dir: where to save the class embeddings
    """
    # H = 1024
    # W = 1280
    os.makedirs(class_embedding_save_dir, exist_ok=True)
    

    # gather all the files
    frame_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(feat_dir) for file in files if files]
    mask_list = [os.path.join(os.path.basename(subdir), file) for subdir, _, files in os.walk(mask_dir) for file in files if files]
    # for mask_name in mask_list:
    #     print("Found mask:", mask_name)

    for n, frame_name in enumerate(frame_list):
        print(f"Processing {n+1}/{len(frame_list)}: {frame_name}")

        # read the frame embeddings
        feat_path = osp.join(feat_dir, frame_name)
        feat = np.load(feat_path)  # shape (H, W, C)
        # feat = feat.cpu().numpy()
        
        # read all the original masks (without any augmentation) of the current frame and organise them into a list
        masks_name = [mask for mask in mask_list if mask.split("_")[0] == frame_name.split(".")[0]] 
        print(f"Found {len(masks_name)} masks for {frame_name}.")
        masks_name = sorted(masks_name)
        
        original_masks = []
        
        for mask_name in masks_name:
            mask_path = osp.join(mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
            # mask = np.uint8(mask == 255)
            mask = np.uint8(mask == 255) * 255    # ensure mask is 0 or 255
            mask = Image.fromarray(mask)
            original_masks.append(mask)
        
        # go through each mask
        for mask, mask_name in zip(original_masks, masks_name):
            # process the mask to the same shape and format as the image
            print("Processing mask:", mask_name)
            zeros = np.zeros_like(mask)
            mask_processed = np.stack((mask, zeros, zeros), axis=-1)
            mask_processed = set_mask(mask_processed)
            mask_processed = F.interpolate(mask_processed, size=torch.Size([64, 64]), mode="bilinear") # shape (1, 3, 64, 64)
            mask_processed = mask_processed.squeeze()[0]
            
            if (True in (mask_processed > 0)) == False:
                # print(f"Skipping {mask_name}: empty mask.")
                # continue
                print(f"Saving empty embedding for {mask_name.replace('.png', '.npy')}")
                class_embedding = np.zeros((feat.shape[-1],), dtype=np.float32)  # shape (256,)
            else:            
                # compute the class embedding using frame SAM feature and processed mask
                class_embedding = feat[mask_processed > 0]
                class_embedding = class_embedding.mean(0).squeeze()
            # if (True in (mask_processed > 0)) == False:
            #     print(f"Skipping {mask_name}: empty mask.")
            #     continue
            # class_embedding = feat[mask_processed > 0]
            # class_embedding = class_embedding.mean(0).squeeze()
            
            # save the class embedding
            save_dir = osp.join(class_embedding_save_dir, mask_name.replace("png","npy"))
            os.makedirs(osp.dirname(save_dir), exist_ok = True) 
            np.save(save_dir, class_embedding)
            # print(f"Saved embedding for {mask_name.replace("png","npy")}")
            embedding_name = mask_name.replace(".png", ".npy")
            print(f"Saved embedding for {embedding_name}")
            

generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/train/0/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/train/0/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/train/0/class_embeddings_h"
)
generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/val/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/val/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold0/val/class_embeddings_h"
)


generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/train/0/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/train/0/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/train/0/class_embeddings_h"
)
generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/val/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/val/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold1/val/class_embeddings_h"
)


generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/train/0/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/train/0/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/train/0/class_embeddings_h"
)
generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/val/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/val/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold2/val/class_embeddings_h"
)


generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/train/0/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/train/0/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/train/0/class_embeddings_h"
)
generate_class_embeddings_from_existing_features(
    mask_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/val/binary_annotations",
    feat_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/val/sam_features_h",
    class_embedding_save_dir="../../data/en17to18_thres0.5/endovis_2018/fold3/val/class_embeddings_h"
)