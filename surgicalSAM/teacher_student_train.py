import sys
sys.path.append("..")
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward import model_forward_function
from copy import deepcopy
import albumentations as A
from tqdm import tqdm
from loss import DiceLoss, kl_bernoulli
from pytorch_metric_learning import losses
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
parser.add_argument('--checkpoint_path', type=str, default=None, help='specify the path to the checkpoint')
args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666  
# data_root_dir = f"../data/{dataset_name}"
# data_root_dir = f"../data/set3/{dataset_name}"
data_root_dir = f"../data/en17to18_thres0.5/{dataset_name}/fold{fold}"
batch_size = 32
# batch_size = 16
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    # num_tokens = 2
    num_tokens = 4 # finetune en17 model
    val_dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                #    map_to_en17 = True, # finetune en17 model using en18 dataset
                                   mode="val",
                                   vit_mode = "h")
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
                                            #  convert2source=True) # finetune en17 model using en18 dataset *convert en18 label to en17
    num_epochs = 100 # finetune en17 model
    lr = 0.0001
    if args.checkpoint_path:
        surgicalSAM_ckp = args.checkpoint_path
    else:
        # surgicalSAM_ckp = f"../ckp/surgical_sam/{dataset_name}/model_ckp.pth"
        # surgicalSAM_ckp = f"./work_dirs/exp1/set2/endovis_2017/{fold}/model_ckp.pth"
        surgicalSAM_ckp = f"/home/users/astar/i2r/stuhuiern/scratch/SurgicalSAM/surgicalSAM/work_dirs/set2_en17_ckp/{fold}/model_ckp.pth"
    
    save_dir = f"./work_dirs/exp10/endovis_2018/Dice_noKLwithAugUnionThres07/{fold}"
    # save_dir = "./work_dirs/set3/endovis_2018/"
    # save_dir = f"./work_dirs/exp4_finetune/thres0.5/endovis_2017_18/{fold}"

elif "17" in dataset_name:
    # num_tokens = 4
    num_tokens = 2 # finetune en18 model
    val_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold, 
                                #    map_to_en18 = True, # finetune en18 model using en17 dataset
                                   vit_mode = "h",
                                   version = 0)
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", fold = fold)
                                            #  convert2source=True) # finetune en18 model using en17 dataset *convert en17 label to en18
    num_epochs = 50 # finetune en18 model
    lr = 0.0001
    if args.checkpoint_path:
        surgicalSAM_ckp = args.checkpoint_path
    else:
        # surgicalSAM_ckp = f"../ckp/surgical_sam/{dataset_name}/fold{fold}/model_ckp.pth"
        surgicalSAM_ckp = f"./work_dirs/exp1/set3/{dataset_name}/model_ckp.pth"
    
    save_dir = f"./work_dirs/endovis_2017/{fold}"
    # save_dir = f"./work_dirs/set3/endovis_2017/{fold}"
    
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


print("======> Load SAM" )
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True

image_encoder = sam_model_registry[f"vit_h"](checkpoint=sam_checkpoint).image_encoder
resizeLongestSide = ResizeLongestSide(image_encoder.img_size)  
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for image encoder via DataParallel")
    image_encoder = torch.nn.DataParallel(image_encoder)
image_encoder.cuda().eval()


print("======> Load Prototypes and Prototype-based Prompt Encoder" )
learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()
checkpoint = torch.load(surgicalSAM_ckp)
protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True
    
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True


# --- 1. Init student (trainable parts) ---
student_prototype_prompt_encoder = protoype_prompt_encoder
student_sam_decoder = sam_decoder
student_learnable_prototypes_model = learnable_prototypes_model

# --- 2. Teacher ---
teacher_protoype_prompt_encoder = deepcopy(protoype_prompt_encoder).eval().cuda()
teacher_sam_decoder = deepcopy(sam_decoder).eval().cuda()
teacher_learnable_prototypes_model = deepcopy(learnable_prototypes_model).eval().cuda()

for p in teacher_protoype_prompt_encoder.parameters():
    p.requires_grad = False
for p in teacher_sam_decoder.parameters():
    p.requires_grad = False
for p in teacher_learnable_prototypes_model.parameters():
    p.requires_grad = False

# --- 3. EMA update ---
@torch.no_grad()
def update_teacher(student, teacher, alpha=0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1-alpha)

# After optimizer.step()
update_teacher(student_prototype_prompt_encoder, teacher_protoype_prompt_encoder)
update_teacher(student_sam_decoder, teacher_sam_decoder)
update_teacher(student_learnable_prototypes_model, teacher_learnable_prototypes_model)

print("======> Define Optmiser and Loss")
optimiser = torch.optim.Adam([
            {'params': student_learnable_prototypes_model.parameters()},
            {'params': student_prototype_prompt_encoder.parameters()},
            {'params': student_sam_decoder.parameters()}
        ], lr = lr, weight_decay = 0.0001)

# --- Define augmentations for student ---
def surgicalsam_transform(
    H=1024, W=1280,
    scale_factor=0.2,
    rotate_angle=30,
    colour_factor=0.4,
):
    combos = [
        A.Compose([A.NoOp()], p=0.5),  # Flip-only 
        A.Compose([
            A.RandomScale(scale_limit=(0, scale_factor), p=1.0),
            A.RandomCrop(height=H, width=W, p=1.0),
        ], p=0.5),
        A.Compose([
            A.Rotate(limit=(0, rotate_angle), p=1.0),
        ], p=0.5),
        A.Compose([
            A.ColorJitter(brightness=colour_factor,
                          contrast=colour_factor,
                          saturation=colour_factor,
                          hue=0, p=1.0),
        ], p=0.5),
    ]

    return A.Compose([
        # A.NoOp()
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf(combos, p=1.0),  # choose one uniformly
    ], additional_targets={"mask2": "mask"}, p=0.5)
    
def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors""" A"""  """ """
    device = x.device
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        
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

print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)

print("======> Start Training and Validation" )
if "18" in dataset_name:
    train_dataset = Endovis18Dataset(data_root_dir = data_root_dir,
                                        mode="train",
                                        vit_mode = vit_mode,
                                        # map_to_en17= True, # finetune en17 model using en18 dataset
                                        with_img = True,
                                        version = 0)
        
elif "17" in dataset_name:
    train_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                        mode="train",
                                        fold = fold,
                                    #  map_to_en18 = True, # finetune en18 model using en17 dataset
                                        vit_mode = vit_mode,
                                        with_img = True,
                                        version = 0)
    
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
best_challenge_iou_val = -100.0
# --- 3. Training loop ---
for epoch in range(num_epochs):   
    # training 
    student_prototype_prompt_encoder.train()
    student_sam_decoder.train()
    student_learnable_prototypes_model.train()
    # count = 0
    for sam_feats, _, cls_ids, masks, _, imgs in tqdm(train_dataloader):
        # if count > 5:
        #     break
        # count += 1
        imgs = imgs.cpu().numpy()
        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()
        pseudolabel_masks = masks.cuda()

        # mask for consistency loss
        with torch.no_grad():
            teacher_prototypes = teacher_learnable_prototypes_model()
            teacher_preds, _ = model_forward_function(teacher_protoype_prompt_encoder, sam_prompt_encoder, teacher_sam_decoder, sam_feats, teacher_prototypes, cls_ids)
            teacher_preds = torch.sigmoid(teacher_preds) # apply sigmoid to get probabilities

        # augment the input image for student and teacher mask
        aug_imgs, aug_teacher_preds, aug_pseudolabels = [], [], []
        T = surgicalsam_transform()
        for i in range(imgs.shape[0]):
            teacher_pred_mask = teacher_preds[i].cpu().numpy().copy()
            pseudolabel_mask = pseudolabel_masks[i].cpu().numpy().copy()
            
            augmented = T(image=imgs[i], mask=teacher_pred_mask, mask2=pseudolabel_mask)
            aug_img = resizeLongestSide.apply_image(augmented['image'])
            aug_img = torch.as_tensor(aug_img, device="cuda")
            aug_img = aug_img.permute(2,0,1).contiguous()
            aug_img = preprocess(aug_img)
            aug_imgs.append(aug_img)

            aug_teacher_mask = torch.from_numpy(augmented['mask']).cuda()        # teacher mask (float probs 0–1)
            aug_pseudolabel_mask = torch.from_numpy(augmented['mask2']).cuda()   # binary 0/255
            
            # threshold teacher mask to binary, then take union with pseudolabel
            aug_union = torch.logical_or(
                # (aug_teacher_mask > 0.5),
                (aug_teacher_mask > 0.7),
                (aug_pseudolabel_mask > 0)
            ).float() * 255.0

            aug_teacher_preds.append(aug_teacher_mask)  # keep teacher prob map as-is
            aug_pseudolabels.append(aug_union)          # binary union mask

        aug_imgs = torch.stack(aug_imgs)
        # print(f"size aug in: {aug_imgs.size()}")
        with torch.no_grad():
            aug_sam_feats = image_encoder(aug_imgs).permute(0, 2, 3, 1)

        aug_teacher_preds = torch.stack(aug_teacher_preds) 
        aug_pseudolabels = torch.stack(aug_pseudolabels) 
        
        aug_pseudolabels = (aug_pseudolabels == 255).to(torch.uint8) * 255
        aug_pseudolabel_class_embeddings = []
        for i, mask in enumerate(aug_pseudolabels):
            feat = aug_sam_feats[i].cpu().numpy()  # SAM feature → NumPy
            zeros = torch.zeros_like(mask)         # stay in Torch
            mask_processed = torch.stack((mask, zeros, zeros), dim=0).unsqueeze(0)  # (1,3,H,W)
            mask_processed = F.interpolate(mask_processed.float(), size=(64,64), mode="bilinear")
            mask_processed = mask_processed.squeeze(0)[0]   # take channel 0

            if not torch.any(mask_processed > 0):
                class_embedding = np.zeros((feat.shape[-1],), dtype=np.float32)
            else:
                class_embedding = feat[mask_processed.cpu().numpy() > 0].mean(0).squeeze()
            aug_pseudolabel_class_embeddings.append(class_embedding)

        student_prototypes = student_learnable_prototypes_model()
        student_preds, _ = model_forward_function(student_prototype_prompt_encoder, sam_prompt_encoder, student_sam_decoder, aug_sam_feats, student_prototypes, cls_ids)
        student_preds = torch.sigmoid(student_preds) # apply sigmoid to get probabilities
        
        # compute loss 
        contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
        seg_loss_model = DiceLoss(activate=False).cuda()
        aug_pseudolabel_class_embeddings = torch.tensor(
            np.stack(aug_pseudolabel_class_embeddings),
            dtype=torch.float32,
            device=student_prototypes.device
        )
        contrastive_loss = contrastive_loss_model(student_prototypes, torch.tensor([i for i in range(1, student_prototypes.size()[0] + 1)]).cuda(), ref_emb = aug_pseudolabel_class_embeddings, ref_labels = cls_ids)
        aug_pseudolabels = aug_pseudolabels/255
        seg_loss = seg_loss_model(student_preds, aug_pseudolabels)
        # kl_bg = (1 - aug_pseudolabels) * kl_bernoulli(aug_teacher_preds, student_preds)
        # kl_bg = kl_bernoulli(aug_teacher_preds, student_preds)
        # kl_bg = kl_bg.mean()
        # alpha = 0.5
        # loss = seg_loss + contrastive_loss + alpha * kl_bg 
        loss = seg_loss + contrastive_loss 
                
        # print_log(f"Training - Epoch: {epoch}/{num_epochs-1}; Loss: {loss.item():.4f}; seg_loss: {seg_loss}; contrastive_loss: {contrastive_loss}; consistency loss: {consistency_loss}", log_file)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # update teacher with EMA
        update_teacher(student_prototype_prompt_encoder, teacher_protoype_prompt_encoder)
        update_teacher(student_sam_decoder, teacher_sam_decoder)
        update_teacher(student_learnable_prototypes_model, teacher_learnable_prototypes_model)

    # validation 
    binary_masks = dict()
    student_learnable_prototypes_model.eval()
    student_prototype_prompt_encoder.eval()
    student_sam_decoder.eval()

    with torch.no_grad():
        prototypes = student_learnable_prototypes_model()
        for sam_feats, mask_names, cls_ids, _, _ in tqdm(val_dataloader):
            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()    

            preds , preds_quality = model_forward_function(student_prototype_prompt_encoder, sam_prompt_encoder, student_sam_decoder, sam_feats, prototypes, cls_ids)    

            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
    print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ", log_file)
    
    if endovis_results["challengIoU"] > best_challenge_iou_val:
        best_challenge_iou_val = endovis_results["challengIoU"]
        
        torch.save({
            'prototype_prompt_encoder_state_dict': student_prototype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': student_sam_decoder.state_dict(),
            'prototypes_state_dict': student_learnable_prototypes_model.state_dict(),
        }, osp.join(save_dir,'model_ckp.pth'))

        print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)        

# python teacher_student_train.py  --dataset endovis_2017  --fold 0 --checkpoint_path /scr/user/huiern214/SurgicalSAM/surgicalSAM/work_dirs/exp1/set2/endovis_2017/0/model_ckp.pth
