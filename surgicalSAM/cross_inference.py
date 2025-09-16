import sys
sys.path.append("..")
from segment_anything import sam_model_registry
import torch 
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from model_forward import model_forward_function
import argparse
from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default="endovis_2018", 
                        choices=["endovis_2018", "endovis_2017"], 
                        help='dataset used for training')
    parser.add_argument('--test_dataset', type=str, default="endovis_2017", 
                        choices=["endovis_2018", "endovis_2017"], 
                        help='dataset used for testing')
    parser.add_argument('--test_dataset_type', type=str, default="val",
                        choices=["train", "val"],
                        help='dataset type for testing, either train or val')
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], 
                        help='specify fold number for endovis_2017 dataset')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='specify the path to the checkpoint')
    parser.add_argument('--dataset_version', type=str, default='set1',
                        help='specify the dataset version')
    return parser.parse_args()


def main():
    args = parse_args()
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    test_dataset_type = args.test_dataset_type
    fold = args.fold
    dataset_version = args.dataset_version
    
    print(f"======> Cross-Dataset Evaluation: {train_dataset} -> {test_dataset}")
    
    # Set data paths
    data_root_dir = f"../data/{dataset_version}/{test_dataset}"
    
    # Load dataset-specific parameters
    if test_dataset == "endovis_2018":
        # num_tokens = 2
        dataset = Endovis18Dataset(data_root_dir=data_root_dir, 
                                   mode=test_dataset_type, # default is "val"
                                   map_to_en17=True,
                                   vit_mode="h")

        gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir,
                                                 convert2source=True,
                                                 mode=test_dataset_type)
                                                # mode="val")
    
    elif test_dataset == "endovis_2017":
        # num_tokens = 4
        dataset = Endovis17Dataset(data_root_dir=data_root_dir, 
                                  mode=test_dataset_type, # default is "val"
                                  fold=fold, 
                                  vit_mode="h",
                                  map_to_en18=True,
                                  version=0)
        
        gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir,
                                                 convert2source=True,
                                                mode=test_dataset_type,
                                                fold=fold)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load SAM
    print("======> Load SAM")
    sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h_no_image_encoder"
    sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_prompt_encoder.cuda()
    sam_decoder.cuda()
    
    # Load SurgicalSAM trained on source dataset
    print(f"======> Load SurgicalSAM trained on {train_dataset}")
    
    # Define the number of classes based on the training dataset
    if train_dataset == "endovis_2018":
        num_classes = 7
        num_tokens_train = 2
        if args.checkpoint_path:
            surgicalSAM_ckp = args.checkpoint_path
        else:
            surgicalSAM_ckp = f"./work_dirs/exp1/{dataset_version}/{train_dataset}/model_ckp.pth"

    elif train_dataset == "endovis_2017":
        num_classes = 7
        num_tokens_train = 4
        if args.checkpoint_path:
            surgicalSAM_ckp = args.checkpoint_path
        else:
            # surgicalSAM_ckp = f"./work_dirs/exp1/{dataset_version}/{train_dataset}/{fold}/model_ckp.pth"
            surgicalSAM_ckp = f"./work_dirs/exp4_finetune/thres0.5_aug/endovis_2017_18/{fold}/model_ckp.pth"

    # Define the models
    learnable_prototypes_model = Learnable_Prototypes(num_classes=num_classes, feat_dim=256).cuda()
    protoype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim=256, 
                                                      hidden_dim_dense=128, 
                                                      hidden_dim_sparse=128, 
                                                      size=64, 
                                                      num_tokens=num_tokens_train).cuda()
    
    # Load the weights
    checkpoint = torch.load(surgicalSAM_ckp)
    protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])
    
    # Set requires_grad to False
    for name, param in sam_prompt_encoder.named_parameters():
        param.requires_grad = False
    for name, param in sam_decoder.named_parameters():
        param.requires_grad = False
    for name, param in protoype_prompt_encoder.named_parameters():
        param.requires_grad = False
    for name, param in learnable_prototypes_model.named_parameters():
        param.requires_grad = False
    
    # Start inference
    print("======> Start Cross-Dataset Inference")
    binary_masks = dict()
    thr = 0 
    
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()
    
    with torch.no_grad():
        prototypes = learnable_prototypes_model()
        
        # swapped mask_names & cls_ids (test 1 2 3 4 -> train 1 2 3 6)
        for sam_feats, mask_names, cls_ids, _, _ in dataloader: # test dataloader
            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()
            
            preds, preds_quality = model_forward_function(
                protoype_prompt_encoder, 
                sam_prompt_encoder, 
                sam_decoder, 
                sam_feats, 
                prototypes, 
                cls_ids 
            )
            
            # Create binary masks
            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)
            
    # Create and evaluate masks
    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
    print(f"\n======> Cross-Dataset Results ({train_dataset} → {test_dataset}):")
    print("Full Results:")
    print(endovis_results)
    
    # Filter results to only include common classes
    # Class name to index mapping
    if train_dataset == "endovis_2018":
        class_map = {
            "BF": 0,
            "PF": 1,
            "LND": 2,
            "MCS": 3
        }
    else:
        class_map = {
            "BF": 0,
            "PF": 1,
            "LND": 2,
            "MCS": 5
        }
    ciou_list = endovis_results["cIoU_per_class"]

    print("IoU per class:")
    for class_name, idx in class_map.items():
        iou = ciou_list[idx]
        print(f"  {class_name}: {iou:.3f}" if not np.isnan(iou) else f"  {class_name}: NaN")

    # Optional: compute mean IoU of selected classes
    selected_values = [ciou_list[i] for i in class_map.values()]
    mean_iou = np.nanmean(selected_values)
    print(f"\nMean IoU (BF, PF, LND, MCS): {mean_iou:.3f}")
    
    # if True:
    #     output_mask_dir = f"output_masks_endovis17_to_endovis18_set2/thres0.5/fold" + str(fold) + "/" + test_dataset_type
    #     # output_mask_dir = f"output_masks_endovis17_to_endovis18_set2/percentile90/fold" + str(fold) + "/" + test_dataset_type
    #     os.makedirs(output_mask_dir, exist_ok=True)

    #     for fullname, mask in endovis_masks.items():
    #         # mask_np = mask.cpu().numpy().astype(np.uint8)  # shape: (H, W)
    #         if isinstance(mask, torch.Tensor):
    #             mask_np = mask.cpu().numpy().astype(np.uint8)
    #         else:
    #             mask_np = mask.astype(np.uint8)
    #         seq_name, name = fullname.split('/')
    #         os.makedirs(os.path.join(output_mask_dir, seq_name), exist_ok=True)
    #         if cv2.imwrite(os.path.join(output_mask_dir, seq_name, name), mask_np):
    #             print(f"Saved mask for {name} to {os.path.join(output_mask_dir, seq_name, name)}")
            # if cv2.imwrite(os.path.join(output_mask_dir, name), mask_np):
            #     print(f"Saved mask for {name} to {os.path.join(output_mask_dir, name)}")

        # # === Your color map ===
        # label2color = {
        #     0: np.array([0, 0, 0]),       # background
        #     1: np.array([255, 0, 0]),     # red
        #     2: np.array([0, 255, 0]),     # green
        #     3: np.array([0, 0, 255]),     # blue
        #     4: np.array([255, 255, 0]),   # yellow
        #     5: np.array([255, 0, 255]),   # magenta
        #     6: np.array([0, 255, 255]),   # cyan
        #     7: np.array([128, 128, 128])  # gray
        # }

        # # === Optional filtering ===
        # active_classes = {1, 2, 3, 6}  # set() of classes to keep; others will be replaced with 0
        # # cover all classes in the dataset
        # # active_classes = {1, 2, 3, 4, 5, 6, 7}  # set() of classes to keep; others will be replaced with 0
        # # === Class name legend ===
        # class_names = {
        #     1: "Bipolar Forceps",
        #     2: "Prograsp Forceps",
        #     3: "Large Needle Driver",
        #     4: "Vessel Sealer",
        #     5: "Grasping Retractor",
        #     6: "Monopolar Curved Scissors",
        #     7: "Others"
        # }

        # def apply_label2color(mask):
        #     color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        #     for label, color in label2color.items():
        #         color_mask[mask == label] = color
        #     return color_mask

        # def overlay_mask_on_image(image, mask, alpha=0.5):
        #     color_mask = apply_label2color(mask)
        #     return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

        # def visualize_and_save_comparison(fullname, original, gt_mask, pred_mask, save_dir):
        #     fig, axs = plt.subplots(1, 5, figsize=(24, 5))

        #     axs[0].imshow(original)
        #     axs[0].set_title("Original")
        #     axs[0].axis('off')
        #     # Add caption below the Original image
        #     axs[0].text(0.5, -0.15, fullname, fontsize=10, ha='center', va='center', transform=axs[0].transAxes)

        #     axs[1].imshow(apply_label2color(gt_mask))
        #     axs[1].set_title("GT Mask")
        #     axs[1].axis('off')

        #     axs[2].imshow(apply_label2color(pred_mask))
        #     axs[2].set_title("Pred Mask")
        #     axs[2].axis('off')

        #     axs[3].imshow(overlay_mask_on_image(original, gt_mask))
        #     axs[3].set_title("Overlay: GT")
        #     axs[3].axis('off')

        #     axs[4].imshow(overlay_mask_on_image(original, pred_mask))
        #     axs[4].set_title("Overlay: Pred")
        #     axs[4].axis('off')

        #     # Add label legend
        #     handles = [
        #         plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
        #                 markerfacecolor=np.array(label2color[i])/255.0, markersize=10)
        #         for i in sorted(active_classes)
        #     ]
        #     fig.legend(handles=handles, loc='lower center', ncol=4, fontsize='small')

        #     plt.tight_layout(rect=[0, 0.05, 1, 1])
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     #  No such file or directory: '/scr/user/huiern214/SurgicalSAM/surgicalSAM/visual_results_en17_to_en18_set2/fold0/compare_seq2/00067.png'
        #     # os.makedirs(save_dir, exist_ok=True)
        #     seq_name, name = fullname.split('/')
        #     os.makedirs(os.path.join(save_dir, seq_name), exist_ok=True)
        #     plt.savefig(os.path.join(save_dir, seq_name, name))
        #     # plt.savefig(os.path.join(save_dir, name))
            
        #     plt.close()

        # # === Example integration ===
        # # with fold 
        # visual_output_dir = "visual_results_en17_to_en18_set2/" + f"/fold{fold}/" + test_dataset_type

        # for name, pred_mask in endovis_masks.items():
        #     # pred_np = pred_mask.cpu().numpy().astype(np.uint8)
        #     if isinstance(pred_mask, torch.Tensor):
        #         pred_np = pred_mask.cpu().numpy().astype(np.uint8)
        #     else:
        #         pred_np = pred_mask.astype(np.uint8)
        #     # gt_np = gt_endovis_masks[name].cpu().numpy().astype(np.uint8)
        #     if isinstance(gt_endovis_masks[name], torch.Tensor):
        #         gt_np = gt_endovis_masks[name].cpu().numpy().astype(np.uint8)
        #     else:
        #         gt_np = gt_endovis_masks[name].astype(np.uint8)

        #     # Mask out classes not in active_classes
        #     pred_np = np.where(np.isin(pred_np, list(active_classes)), pred_np, 0)
        #     gt_np = np.where(np.isin(gt_np, list(active_classes)), gt_np, 0)

        #     # endovis18 /scr/user/huiern214/SurgicalSAM/data/set1/endovis_2018/val/images
        #     # image_path = os.path.join(data_root_dir, "val/images", name)
        #     # image_path = os.path.join(data_root_dir, test_dataset_type, "images", name)
        #     if test_dataset == "endovis_2018":
        #         if test_dataset_type == "val":
        #             image_path = os.path.join(data_root_dir, test_dataset_type, "images", name)
        #         elif test_dataset_type == "train":
        #             image_path = os.path.join(data_root_dir, test_dataset_type, "0", "images", name)
        #     elif test_dataset == "endovis_2017":
        #         image_path = os.path.join(data_root_dir, "0", "images", name) # apply all sequences 

        #     image = cv2.imread(image_path)
        #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     visualize_and_save_comparison(name, image_rgb, gt_np, pred_np, visual_output_dir)
    
if __name__ == "__main__":
    main()