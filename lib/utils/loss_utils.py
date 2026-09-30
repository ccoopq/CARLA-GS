#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from math import exp
from lib.utils.img_utils import save_img_torch
from lib.config import cfg


def l1_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''

    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]

    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]
    
    loss = ((torch.abs(network_output - gt))).mean()

    return loss

def l2_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''
    
    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]    
    
    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]

    loss =  (((network_output - gt) ** 2)).mean()

    return loss

    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    '''
    img1, img2: (C, H, W)
    mask: (1, H, W)
    '''    
    
    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    
    if mask is not None:
        mask = mask.squeeze(0)
        img1 = img1[mask]
        img2 = img2[mask]
    
    # mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(dim=0, keepdim=True)    
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
    
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if mask is not None:
        img1 = torch.where(mask, img1, torch.zeros_like(img1))
        img2 = torch.where(mask, img2, torch.zeros_like(img2))
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

import lpips as LPIPS
import torch
import torch.nn.functional as F
lpips_model = LPIPS.LPIPS(net='alex')  # or 'vgg'
if torch.cuda.is_available():
    lpips_model = lpips_model.cuda()
lpips_model.eval()  

def lpips(img1, img2, mask=None):
    """
    img1, img2: (C, H, W)   range: [0,1]
    mask: (1, H, W) optional

    return: scalar LPIPS
    """

    # LPIPS expects (N,3,H,W) and range [-1,1]
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # convert to [-1,1]
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    if mask is not None:
        # mask → (1,1,H,W)
        mask = mask.unsqueeze(0)

        img1 = img1 * mask
        img2 = img2 * mask

    # LPIPS forward
    loss = lpips_model(img1, img2)

    return loss.mean()

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img


def compute_flat_loss(gaussians, point_mask=None):
    scales = gaussians.get_scaling
    if point_mask is not None:
        if point_mask.dtype != torch.bool:
            point_mask = point_mask.bool()
        if point_mask.ndim == 1 and point_mask.shape[0] == scales.shape[0]:
            scales = scales[point_mask]
        else:
            pass
    if scales.numel() == 0:
        return scales.new_tensor(0.0)
    min_axis, _ = torch.min(scales, dim=1)  # (M,)
    return min_axis.mean()

def patch_based_ncc_loss(pred, gt, mask, patch_size=3, stride=1):
    H, W = pred.shape
    # Add batch and channel dimensions: (1, 1, H, W)
    pred = pred * mask 
    gt = gt * mask

    pred = pred.float().unsqueeze(0).unsqueeze(0)
    gt = gt.float().unsqueeze(0).unsqueeze(0)
    mask = mask.float().unsqueeze(0).unsqueeze(0)

    unfold = torch.nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=stride)
    pred_patches = unfold(pred)[0].permute(1,0)  # Shape: (1, K*K, h*w)
    gt_patches = unfold(gt)[0].permute(1,0)      # Shape: (1, K*K, h*w)

    # Subtract the mean from predictions and ground truth
    pred_centered = pred_patches - pred_patches.mean()
    gt_centered = gt_patches - gt_patches.mean()

    # Calculate standard deviations of centered predictions and ground truth
    pred_std = torch.sqrt(torch.mean(pred_centered ** 2, dim=1))
    gt_std = torch.sqrt(torch.mean(gt_centered ** 2, dim=1))

    # Calculate the NCC
    ncc = torch.sum(pred_centered * gt_centered, dim=1) / (pred_std * gt_std + 1e-5)
    ncc_loss = 1 - (ncc / patch_size**2)
    return ncc_loss.mean()


def angular_loss(pred_normals, gt_normals, mask=None):
    """
    Computes the angular loss between predicted and ground truth normals.
    
    Args:
        pred_normals (torch.Tensor): Predicted normals of shape (3, H, W).
        gt_normals (torch.Tensor): Ground truth normals of shape (3, H, W).
        mask (torch.Tensor, optional): Binary mask of shape (H, W). Defaults to None.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    pred_normals = F.normalize(pred_normals, p=2, dim=0)
    gt_normals = F.normalize(gt_normals, p=2, dim=0)

    cosine_similarity = torch.sum(pred_normals * gt_normals, dim=0)  # Shape: (H, W)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    loss = 1 - cosine_similarity  # Shape: (H, W)
    
    if mask is not None:
        loss = loss * mask  # Apply mask to the loss
    
    if mask is not None:
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss

def shift_right(pred_normals):
    pred_normals_right = F.pad(pred_normals[:, :, :-1], (1, 0), mode='replicate')  # (3, H, W)
    return pred_normals_right

def shift_down(pred_normals):
    pred_normals_down = F.pad(pred_normals[:, :-1, :], (0, 0, 1, 0), mode='replicate')  # (3, H, W)
    return pred_normals_down

def geo_consist_loss(pred_normals, depth, mask=None):
    pred_normals = F.normalize(pred_normals, p=2, dim=0)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    depth = depth.unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    epsilon = 1e-6 
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    grad_magnitude_normalized = 1.0 - (grad_magnitude - grad_min) / (grad_max - grad_min + epsilon)
    grad_magnitude_normalized = grad_magnitude_normalized.squeeze(0).squeeze(0)  # Shape: (H, W)
    
    if mask is not None:
        grad_magnitude_normalized = grad_magnitude_normalized * mask  # Shape: (H, W)
   
    pred_normals_right = shift_right(pred_normals)
    pred_normals_down = shift_down(pred_normals)
    dot_right = torch.sum(pred_normals * pred_normals_right, dim=0)
    dot_down = torch.sum(pred_normals * pred_normals_down, dim=0)
    
    dot_right = torch.clamp(dot_right, -1.0, 1.0)
    dot_down = torch.clamp(dot_down, -1.0, 1.0)
    
    angle_right = 1 - dot_right  # Shape: (H, W)
    angle_down = 1 - dot_down    # Shape: (H, W)
    angle_loss = (angle_right + angle_down) / 2.0  # Shape: (H, W)
    weighted_loss = angle_loss * grad_magnitude_normalized  # Shape: (H, W)
    loss = weighted_loss.sum() / (grad_magnitude_normalized.sum() + epsilon)
    
    return loss

def geo_consist_loss_3x3(pred_normals, depth, mask=None):
    pred_normals = F.normalize(pred_normals, p=2, dim=0)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    depth = depth.unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    epsilon = 1e-6
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    grad_magnitude_normalized = 1.0 - (grad_magnitude - grad_min) / (grad_max - grad_min + epsilon)
    grad_magnitude_normalized = grad_magnitude_normalized.squeeze(0).squeeze(0)  # (H, W)

    if mask is not None:
        grad_magnitude_normalized = grad_magnitude_normalized * mask  # (H, W)

    # pred_normals: (3, H, W) -> (1, 3, H, W)
    pred_normals_4d = pred_normals.unsqueeze(0)
    avg_normals = F.avg_pool2d(pred_normals_4d, kernel_size=3, stride=1, padding=1)  # (1,3,H,W)
    avg_normals = F.normalize(avg_normals.squeeze(0), p=2, dim=0)  # (3, H, W)

    dot = torch.sum(pred_normals * avg_normals, dim=0)  # (H, W)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle_loss = 1 - dot  # (H, W)

    weighted_loss = angle_loss * grad_magnitude_normalized
    loss = weighted_loss.sum() / (grad_magnitude_normalized.sum() + epsilon)

    return loss