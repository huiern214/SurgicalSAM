from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 map_to_en17 = False,
                 version = 0):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
        self.map_to_en17 = map_to_en17
        
        # Mapping from Endovis2018 to Endovis2017 if needed
        self.mapping = {
            1: 1,  # BF
            2: 2,  # PF
            3: 3,  # LND
            4: 6,  # MCS (EN18: class 4 → EN17: class 6)
            5: 5,
            6: 4,
            7: 7
        }
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            
            if map_to_en17:
                # only class id 1, 2, 3, 6 (en17 model train/predict on en18 dataset)
                self.mask_list += [osp.join(osp.basename(subdir), i) for i in files if re.search(r"class[1236]", i)]
            else:
                # all classes
                self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]
            # if en18 model train for 4 classes only
            # self.mask_list += [osp.join(osp.basename(subdir), i) for i in files if re.search(r"class[1234]", i)]       
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        # Map class id if requested
        if self.map_to_en17:
            cls_id = self.mapping.get(cls_id, -1)  # default to -1 if not found
            # update the class ID in the mask name string too
            mask_name = re.sub(r"class\d+", f"class{cls_id}", mask_name)
            
        return sam_feat, mask_name, cls_id, mask, class_embedding
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 map_to_en18 = False,
                 version = 0):
                        
        self.vit_mode = vit_mode
        self.map_to_en18 = map_to_en18
        
        # Mapping from Endovis2017 to Endovis2018 if needed
        self.mapping = {
            1: 1,  # BF
            2: 2,  # PF
            3: 3,  # LND
            6: 4,  # MCS (EN17: class 6 → EN18: class 4)
            4: 6,  # VS → UP
            5: 5,  # GR → SI
            7: 7   # OT → CA
        }
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            if self.map_to_en18:
                # only class id 1, 2, 3, 4 (en18 model train/predict on en17 dataset)
                self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path) if re.search(r"class[1234]", mask)]
            else:
                # all classes
                self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            # if en17 model train for 4 classes only
            # self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path) if re.search(r"class[1236]", mask)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        # Map class id if requested
        if self.map_to_en18:
            cls_id = self.mapping.get(cls_id, -1)  # default to -1 if not found
            # update the class ID in the mask name string too
            mask_name = re.sub(r"class\d+", f"class{cls_id}", mask_name)
        
        return sam_feat, mask_name, cls_id, mask, class_embedding
    
