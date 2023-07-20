
"""
Simple NeRF training code to test GeNVS implementation

Modified dataloader only loads from single object, with first view of each batch being the same in all cases.

"""
from omegaconf import DictConfig, OmegaConf
#import hydra

import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import  AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from NERFdataset import dataset
from nerf.utils import render_multi_view
from nerf_model import Nerf
import torch.nn as nn

from collections import defaultdict


import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

output_dir = 'output_tmp/'
os.makedirs(output_dir, exist_ok=True)

@torch.no_grad()
def sample(model, data):
    """
    Performs sampling from the model during inference.

    Parameters:
    model (nn.Module): The model to perform sampling from.
    data (dict): A dictionary containing the data to sample.
    nv (int): Number of views to sample.

    Returns:
     tuple containing targets, first view, depth maps, and samples.
    """
    img = data['imgs']
    targets = 0.5*(img+1)
    
    B, V = targets.shape[:2]

    # Downsample targets


    poses = data['poses']#.cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k']#.cuda()
    camera_d = data['camera_d']#.cuda()
    #print('intrinsics', intrinsics.shape)


    cameras  = (camera_k, camera_d, poses[:,:1])

    Q = 4

    render_poses = poses[:,:Q]    

    img = data['imgs']#.cuda()
    targets = 0.5*(img+1)
        
    B, V = targets.shape[:2]

    # Downsample targets

    poses = data['poses']#.cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k']#.cuda()
    camera_d = data['camera_d']#.cuda()
        
    cameras  = (camera_k, camera_d, poses[:,:1])

    render_poses = poses[:,:Q]

    first_view, d1, o1 = render_multi_view(model.nerf, render_poses, intrinsics[0], model.triplanes, cameras)
    first_view_rgb = first_view[:, :, :3]
    d1 = d1.expand(-1,-1,3,-1,-1)
    o1 = o1.expand(-1,-1,3,-1,-1)

    return targets, first_view_rgb, d1, o1


def stack_img(v):
    """
    Combine multiple images

    Parameters:
    v (array, BQCHW)

    Returns:
    v with channels moved to last index, and shape B, Q*H, W, C
    """
    v = v.cpu().clip(0,1).numpy().transpose(0,1,3,4,2)
    v = v.reshape(v.shape[0], v.shape[1]*v.shape[2], v.shape[3], v.shape[4])
    return v


def train(cfg, device='cpu'):
    """
    Trains a Neural Radiance Fields (NeRF) model

    Parameters:
    rank (int): Rank of the current process.
    world_size (int): Total number of processes.
    cfg (dict): Configuration dictionary
    """

    transfer = cfg["transfer"] #(str): Path to a pretrained model if transfer learning is to be performed. Default is "".
    use_wandb = cfg["use_wandb"] #(bool): Whether to use Weights & Biases for logging and visualization. Default is False.

    if use_wandb and rank==0:
        wandb.init(
        entity=None,
        project="genvs",
        job_type="train",
	    )
    
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")


    # ------------ Init
    step = 0
    num_epochs = cfg.max_epochs
    image_size = cfg.image_size
    batch_size = cfg.batch_size
    acc_steps = cfg.gradient_accumulation_steps
    n_sample =8 # cfg.sample_every_n_steps
    grow = cfg.grow
    combine = cfg.combine
    use_amp = False
    
    epochs_plot_loss = cfg.epochs_checkpoint
    epochs_plot_loss2 = cfg.epochs_checkpoint_save
    load_optim = cfg.load_optim_state_from_checkpoint

    if use_wandb and rank==0:
        wandb.config.update(
	        {
            "train_batch_size":batch_size,
            }
        )

    d = dataset('train', imgsize=image_size)
    
    loader = DataLoader(d, batch_size=batch_size, drop_last=True)

    
    # Model setting
    nerf_diff_params = cfg.nerf_diff
    model = Nerf(**nerf_diff_params)

    
    optimizer = AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer == "":
        step = 0
        start_epoch = 0
    else:
        print('resume from: ', transfer)
        transfer_filename = cfg['transfer_filename']
        ckpt = torch.load(os.path.join(transfer, transfer_filename))

        model.module.load_state_dict(ckpt['model'])
        if load_optim:
            optimizer.load_state_dict(ckpt['optim'])

        step = ckpt['step']
        start_epoch = ckpt['epoch']+1   

        del ckpt


    t00 = t0 = time.monotonic()
    # Training loop
    for e in range(start_epoch, num_epochs):

        print(f'starting epoch {e}')
        
        model.train()
        
        print('LR', [p['lr'] for p in optimizer.param_groups])
        lt = time.time()

        # For each sample in the dataset
        pbar = tqdm(loader)

        epoch_loss = 0.0
        running_loss = 0.0
        running_loss_rgb = 0.0
        epoch_loss_details = defaultdict(float)

        for data in pbar:

            loss, loss_details = model(data)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss
            for d in loss_details:
                epoch_loss_details[d] += loss_details[d].item()

            running_loss += loss.item()



            if step % n_sample == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'loss : {running_loss_rgb/n_sample} {running_loss/n_sample} {lr}')
                if use_wandb:
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": step,
                            "train/epoch": e,
                            **{f"train/{k}": v for k, v in loss_details.items()},
            			}
                    )

            
                running_loss = 0.0
                running_loss_rgb = 0.0
                formatted_images = []
                img1, v1, d1, o1 = sample(model, data)

                img1 = stack_img(img1)
                v1 = stack_img(v1)
                d1 = stack_img(d1)
                o1 = stack_img(o1)
                    
                for k in range(len(img1)):

                    output = np.concatenate((img1[k], v1[k], d1[k], o1[k]), axis=1)
                    output = (255*np.clip(output,0,1)).astype(np.uint8)
                    imwrite(f'{output_dir}/full-{step:06d}-{k}.png', output)


            step += 1
#        scheduler.step()
            
        print('loss epoch', epoch_loss / len(loader))
        epoch_loss_details = {k:v/len(loader) for k,v in epoch_loss_details.items()}
        print('loss epoch', epoch_loss_details)

        # Epoch checkpoint save
        if e % epochs_plot_loss == 0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"{output_dir}/large-k-multi-latest.pt")

    
#@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    #n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    #world_size = n_gpus
    #mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)
    train(cfg)

if __name__ == "__main__":
    cfg = OmegaConf.load('config/config_nerf.yaml')
    main(cfg)
