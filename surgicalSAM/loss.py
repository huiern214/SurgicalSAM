import torch 
import torch.nn.functional as F
import torch.nn as nn 

# Code borrowed from MMSegmentation https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/dice_loss.py

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss,
                       weight = None,
                       reduction = 'mean',
                       avg_factor = None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()


    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)
   
    
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        
    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)
      
        return loss


def kl_bernoulli(p, q, eps=1e-6):
    """
    Compute Bernoulli KL divergence: KL(p || q)
    Using PyTorch's F.kl_div for numerical stability.
    
    Args:
        p: [B, H, W] - target distribution (teacher predictions)
        q: [B, H, W] - input distribution (student predictions)
        eps: small constant for numerical stability
    
    Returns:
        [B, H, W] - per-pixel KL divergence
    """
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    
    # Stack into 2-class distributions [B, 2, H, W]
    p2 = torch.stack([p, 1 - p], dim=1) # target (teacher)
    q2 = torch.stack([q, 1 - q], dim=1) # input (student)
    
    # F.kl_div(input_log_probs, target_probs) computes KL(target || input)
    # So F.kl_div(q2.log(), p2) computes KL(p || q)
    kl = F.kl_div(q2.log(), p2, reduction="none").sum(dim=1)  # [B, H, W]
    
    return kl

# def bce_loss(
#     probs,        # after sigmoid: [B,1,H,W] or [B,H,W]
#     target,         # same shape, values 0 or 1
#     per_pixel = False,      # True → return per-pixel loss map; False → return mean scalar
#     weight = None,
#     eps = 1e-7
# ):
#     """
#     BCE loss for sigmoid outputs in segmentation (no ignore_index).
    
#     Args:
#         probs: model output after sigmoid.
#         target: binary ground truth (0/1).
#         per_pixel: if True, returns per-pixel loss map.
#         weight: optional weighting tensor, same shape as probs or broadcastable.
#         eps: small constant for numerical stability.
#     """
#     # clamp to avoid log(0)
#     probs = probs.clamp(eps, 1 - eps)
#     target = target.to(probs.dtype)

#     # element-wise BCE (no reduction)
#     loss = F.binary_cross_entropy(probs, target, reduction="none")

#     if weight is not None:
#         loss = loss * weight

#     # pixel map or scalar
#     if per_pixel:
#         return loss
#     else:
#         return loss.mean()

# def dice_loss_with_consistency(p_student, p_teacher_aug, offline_pseudolabel_mask):
#     """
#     Compute consistency loss with offline pseudo label as mask.
    
#     Formula:
#         loss = [offline_pseudolabel_mask * Dice_offline + (1-offline_pseudolabel_mask) * KL_online].mean()
    
#     Interpretation:
#         - Where offline_pseudolabel_mask > 0.5 (foreground): use offline pseudo label
#         - Where offline_pseudolabel_mask < 0.5 (background): use online teacher label
    
#     Args:
#         p_student: [B, H, W] - student predictions (probabilities)
#         p_teacher_aug: [B, H, W] - teacher predictions on augmented image (online label)
#         offline_pseudolabel_mask: [B, H, W] - offline pseudo labels after augmentation (0-1)
#                                This also serves as the mask
    
#     Returns:
#         scalar loss
#     """
#     # print(f"p_student {p_student.shape}, p_teacher_aug {p_teacher_aug.shape}, offline pseudolabel {offline_pseudolabel_mask.shape}")
#     # Compute both KL divergences per-pixel
#     # bce_offline = bce_loss(p_student, offline_pseudolabel_mask, per_pixel=True)  # [B, H, W]
#     kl_online = kl_bernoulli(p_teacher_aug, p_student)   # [B, H, W]
    
#     # Weighted combination: y_hat acts as mask
#     # Foreground (offline_pseudolabel_mask≈1): use offline, Background (offline_pseudolabel_mask≈0): use online
#     # loss_combined = offline_pseudolabel_mask * bce_offline + (1 - offline_pseudolabel_mask) * kl_online
#     # print(f"offline bce {bce_offline.shape} and online kl {kl_online.shape} and fg pixel {offline_pseudolabel_mask.numel()} and bg pixel {(1-offline_pseudolabel_mask).numel()}")
#     # loss_combined = kl_online
# kl_bg = (1 - offline_pseudolabel_mask) *  kl_bernoulli(p_teacher_aug, p_student)
# kl_bg = kl_bg.mean()
#     # Sum and divide by total number of pixels
#     # total_pixels = loss_combined.numel()
#     # total_pixels = kl_combined.numel()
#     # loss = kl_combined.sum() / total_pixels
    
#     # loss = loss_combined.mean()

#     # print(f"here {loss_combined.sum() / total_pixels} same as mean {loss}")
    
#     return loss