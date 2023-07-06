
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf.network_multi import NeRFNetwork
from nerf.utils import render_multi_view

from fastai.vision.models import resnet34
from fastai_unet import DynamicUnet

import random
from k_pipeline import KPipeline
from k_diffusion.augmentation import KarrasDiffAugmentationPipeline

def make_input_model(size=128, out_dim=3*48):
    """
    Function to make a dynamic U-Net model for input image.

    Parameters:
    size : int, optional
        The size of the input image.

    out_dim : int, optional
        The output dimensions.

    Returns:
    model : nn.Module
        The created model.
    """
    m = resnet34()
    m = nn.Sequential(*list(m.children())[:-2])
    model = DynamicUnet(m, out_dim, (size, size), norm_type=None)
    return model

class NerfDiff(nn.Module):
    """
    A NeRF-Diffusion model for processing multi-view input images.

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
                 color_feat_dim=16,
                 depth_size=64,
                 a_prob=0.1,
                 diff_augmentation_prob=0.2,
                 k_diffusion_config='k_configs/config_upscale_less_noisy.json',
                 lambda_rgb_first=0.01,
                 lambda_rgb_other=1.0,
                 lambda_opacity=0.001,
                 lambda_diffuse=1.0,
                 no_cond_prob=0.1):

        super().__init__()
        self.input_unet = make_input_model(size=input_size, out_dim=color_feat_dim*depth_size)
        self.nerf = NeRFNetwork(color_feat_dim=color_feat_dim) # Always render at 64x64 (but 16 channels)
        self.ddpm_pipeline = KPipeline(k_diffusion_config) #KPipeline('configs/config_upscale_noisy.json')
        self.diff_aug = KarrasDiffAugmentationPipeline(a_prob = a_prob)
        self.diff_augmentation_prob = diff_augmentation_prob
        self.lambda_rgb_first = lambda_rgb_first # Loss weight for rgb rendering of NeRF for first (projected) image
        self.lambda_rgb_other = lambda_rgb_other # Loss weight for rgb rendering of NeRF from novel view
        self.lambda_opacity = lambda_opacity # Loss weight penalizing rendering opacity
        self.lambda_diffuse = lambda_diffuse # Loss weight for diffusion objective
        self.no_cond_prob= no_cond_prob # Probability of replacing the rendered NeRF with normally distributed noise,
                              # and adding a learnt vector to the timestep embedding of the
                              # denoising UNet.

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
        img = data['imgs'].cuda()
        targets = 0.5*(img+1)
        
        B, V = targets.shape[:2]

        if self.train_diffusion_resolution == 64:
            # Downsample targets
            targets= F.interpolate(targets.view(B*V, *targets.shape[2:]), scale_factor = 0.5).view(B, V,3, 64, 64)
            
            
    
        nv = random.randint(1,2)

        # Augment input images with noise?
        
        imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
        img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) # 0 for orig xkpt

        B, V = img_tp.shape[:2]
        img_tp = img_tp.view(B*V,*img_tp.shape[2:])
        # Input augmentation (after normalization our images roughly range from -2 to 2, greater range than that in original paper).
        img_tp = img_tp + ((torch.rand([B*V], device=img_tp.device)>0.5)*torch.rand([B*V], device=img_tp.device))[:,None,None,None]*torch.randn_like(img_tp)
        triplanes = self.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
        triplanes = triplanes.view(B, V, *triplanes.shape[1:])[:,:nv].contiguous()

        # Downsample targets

        poses = data['poses'].cuda()
        intrinsics = data['intrinsics']
        camera_k = data['camera_k'].cuda()
        camera_d = data['camera_d'].cuda()

        cameras  = (camera_k, camera_d, poses[:,:nv])

        Q = 2

        render_poses = torch.cat([poses[:,:(Q-1)], poses[:,-1:]], dim=1)

        first_view, d1, o1 = render_multi_view(self.nerf, render_poses, intrinsics[0], triplanes, cameras)
        


        if self.train_diffusion_resolution == 128:
            first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
            first_view = first_view.view(B, Q, *first_view.shape[1:])
            first_view_rgb = first_view[:, :, :3]
            d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
            d1 = d1.view(B, Q, *d1.shape[1:])
            o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
            o1 = o1.view(B, Q, *o1.shape[1:])
        else: # train diffusion model at resolution 64
            d1 = d1.expand(-1,-1,3,-1,-1)
            o1 = o1.expand(-1,-1,3,-1,-1)


        loss_rgb = self.lambda_rgb_first*((first_view_rgb[:,0] - targets[:,0:1])**2).mean() + self.lambda_rgb_other*((first_view_rgb[:,-1] - targets[:,-1:])**2).mean() 
        loss_opacity =self.lambda_opacity*(o1.mean())

        # Denoise target image, guided by multi-channel rendering

        if self.lambda_diffusion:
            cond_flag = 0 if random.random()>self.no_cond_prob else 1 # 10% chance of conditioning being replaced by noise
        
            if self.diff_augmentation_prob and random.random() > self.diff_augmentation_prob:
                # Differentiable aug - apply to both targets and rendering
                mat, cond = self.diff_aug.get_mat_cond(first_view[:,-1])
                target, _, _ = self.diff_aug.apply(targets[:,-1], mat, cond)
                image, _, _ = self.diff_aug.apply(first_view[:,-1], mat, cond)

                #Diffusion reconstruction loss
                loss_train_diffusion = self.lambda_diffusion*self.ddpm_pipeline.train_step(target, image,  cond_flag=cond_flag, aug_cond=cond)
            else:
                mat, cond = self.diff_aug.get_mat_cond(first_view[:,0])
                target, _, _ = self.diff_aug.apply(targets[:,0], mat, cond)
                image, _, _ = self.diff_aug.apply(first_view[:,0], mat, cond)
            #Diffusion reconstruction loss
            loss_train_diffusion = self.lambda_diffusion*self.ddpm_pipeline.train_step(target, image, cond_flag=cond_flag, aug_cond=cond)

        else:
            loss_train_diffusion = 0.0
	# Combined loss
        loss = loss_train_diffusion +loss_rgb + loss_opacity 

        loss_details = { 'rgb': loss_rgb, 'opacity': loss_opacity, 'diffusion': loss_train_diffusion}

        return loss, loss_details

