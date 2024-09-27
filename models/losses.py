import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

#
# def calculate_segmentation_loss(pred, target, smooth=1.):
#     pred = pred.contiguous()
#     target = target.contiguous()
#
#     intersection = (pred * target).sum(dim=2).sum(dim=2)
#
#     loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
#
#     return loss.mean()


def calculate_segmentation_loss(pred, target, bce_weight=0.5):
    # print(pred.shape)
    # print(target.shape)
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def calculate_landmark_loss(criterion, pred_heatmaps, gt_heatmaps, base_number):

    loss = criterion(pred_heatmaps, gt_heatmaps)
    ratio = torch.pow(base_number, gt_heatmaps)
    loss = torch.mul(loss, ratio)
    loss = torch.mean(loss)

    return loss
