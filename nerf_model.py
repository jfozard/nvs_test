
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf.network_multi import NeRFNetwork
from nerf.utils import render_multi_view

import random


class Nerf(nn.Module):
    """
    A NeRF model for processing multi-view input images.

    Parameters:
    input_size : int, optional
        The size of the input image.

    color_feat_dim : int, optional
        The dimension of color features.

    depth_size : int, optional
        The size of the depth field.

    a_prob : float, optional
        The probability for the KarrasDiffAugmentationPipeline.

    diff_augmentation_prob : float, optional
        The probability for applying diffusion augmentation.

    k_diffusion_config : str, optional
        The path to the configuration file for KPipeline.

    lambda_rgb_first : float, optional
        The loss weight for RGB rendering of NeRF for the first (projected) image.

    lambda_rgb_other : float, optional
        The loss weight for RGB rendering of NeRF from a novel view.

    lambda_opacity : float, optional
        The loss weight penalizing rendering opacity.

    lambda_diffuse : float, optional
        The loss weight for diffusion objective.

    no_cond_prob : float, optional
        The probability of replacing the rendered NeRF with normally distributed noise,
        and adding a learnt vector to the timestep embedding of the denoising UNet.
    """    
    
    def __init__(self,
                 input_size=128,
                 train_diffusion_resolution=128,
                 color_feat_dim=16,
                 depth_size=64,
                 a_prob=0.1,
                 diff_augmentation_prob=0.2,
                 k_diffusion_config='k_configs/config_upscale_less_noisy.json',
                 lambda_rgb_first=0.01,
                 lambda_rgb_other=1.0,
                 lambda_opacity=0.0,
                 lambda_diffusion=1.0,
                 no_cond_prob=0.1):

        super().__init__()
        self.nerf = NeRFNetwork(color_feat_dim=color_feat_dim) # Always render at 64x64 (but 16 channels)
        self.triplanes = nn.Parameter(torch.zeros([1,1,16*64,64,64]).normal_()) 
        self.lambda_rgb=1.0
        self.lambda_opacity=0.01

    def forward(self, data):
        """
        Forward pass of the model.

        Parameters:
        data : dict
            A dictionary containing the input images, poses, camera parameters, etc.

        Returns:
        loss : float
            The combined loss from different stages of the model.

        loss_details : dict
            A dictionary containing the detailed loss from each stage.
        """
        
        img = data['imgs']#.cuda()
        targets = 0.5*(img+1)
        
        B, V = targets.shape[:2]

        # Downsample targets

        poses = data['poses']#.cuda()
        intrinsics = data['intrinsics']
        camera_k = data['camera_k']#.cuda()
        camera_d = data['camera_d']#.cuda()
        
        cameras  = (camera_k, camera_d, poses[:,:1])

        render_poses = poses[:,1:]

        first_view, d1, o1 = render_multi_view(self.nerf, render_poses, intrinsics[0], self.triplanes, cameras)
        first_view_rgb = first_view[:, :, :3]
        d1 = d1.expand(-1,-1,3,-1,-1)
        o1 = o1.expand(-1,-1,3,-1,-1)


        loss_rgb = self.lambda_rgb*((first_view_rgb - targets[:,1:])**2).mean() 
        loss_opacity =self.lambda_opacity*(o1.clip(0, 0.1)/0.1).mean()

        loss = loss_rgb + loss_opacity 

        loss_details = { 'rgb': loss_rgb, 'opacity': loss_opacity }

        return loss, loss_details

    def get_point_set(self, data, resolution=32):
        lin = torch.linspace(-1, 1, resolution)
        grid = torch.stack(torch.meshgrid([lin, lin, lin]), dim=-1)  # Creates a 64x64x64x3 grid
        grid = grid.reshape(-1, 3)  # Flattens to a (64^3)x3 tensor

        grid = grid[None,None,None]
        
        # Sample the NeRF at every point in the grid

        poses = data['poses']#.cuda()
        intrinsics = data['intrinsics']
        camera_k = data['camera_k']#.cuda()
        camera_d = data['camera_d']#.cuda()
        
        cameras  = (camera_k, camera_d, poses[:,:1])

        sigma, color, _ = self.nerf(grid, None, self.triplanes, cameras)

        return (sigma, color), grid
