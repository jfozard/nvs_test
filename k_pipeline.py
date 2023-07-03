#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path

import accelerate
import torch
from torch import nn, optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import trange, tqdm

import k_diffusion as K


class KPipeline(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        self.config = K.config.load_config(open(config_file))
        model_config = self.config['model']
        print(model_config)
        model_config['unet_cond_dim'] = 16
        self.size = model_config['input_size']


        self.inner_model = K.config.make_model(self.config)


        print('Parameters:', K.utils.n_params(self.inner_model))


        self.tf = transforms.Compose([
            transforms.Resize(self.size[0], interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.size[0]),
            K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob']),
        ])


        self.sigma_min = model_config['sigma_min']
        self.sigma_max = model_config['sigma_max']
        self.sample_density = K.config.make_sample_density(model_config)

        self.model = K.config.make_denoiser_wrapper(self.config)(self.inner_model)
        #model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)


    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def sample(self, cond, sampling_timesteps=None):
        x = torch.randn([cond.shape[0], self.config['model']['input_channels'], self.size[0], self.size[1]], device=cond.device) * self.sigma_max
        sigmas = K.sampling.get_sigmas_karras(50 if sampling_timesteps is None else sampling_timesteps, self.sigma_min, self.sigma_max, rho=7., device=cond.device)
        x_0 = K.sampling.sample_dpmpp_2m(self.model, x, sigmas, extra_args={'unet_cond':cond})
        print(x_0.min(), x_0.max())
        return x_0.clip(-1.0,1.0)
    
    @torch.no_grad()
    def sample_all(self, cond, sampling_timesteps=None):
        x = torch.randn([cond.shape[0], self.config['model']['input_channels'], self.size[0], self.size[1]], device=self.device) * self.sigma_max
        sigmas = K.sampling.get_sigmas_karras(50 if sampling_timesteps is None else sampling_timesteps, self.sigma_min, self.sigma_max, rho=7., device=cond.device)
        progress = []
        def callback(u):
            progress.append(u['denoised'].clip(-1.0,1.0).cpu())
        x_0 = K.sampling.sample_dpmpp_2m(self.model, x, sigmas, extra_args={'unet_cond':cond}, callback=callback)
        print(x_0.min(), x_0.max())
        progress.append(x_0.clip(-1.0,1.0))
        return progress

    def train_step(self, reals, cond, cond_flag=0, aug_cond=None):
        # Augmentation step with conditioning
        #reals, _, aug_cond = batch[image_key]

        #
        noise = torch.randn_like(reals)
        if cond_flag:
            cond = torch.rand_like(cond)
        sigma = self.sample_density([reals.shape[0]], device=reals.device)
        losses = self.model.loss(reals, noise, sigma, unet_cond=cond, aug_cond=aug_cond)
        loss = losses.mean()
        return loss
   


#a = DDPMPipeline()

