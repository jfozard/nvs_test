
import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import time
import argparse

from einops import rearrange
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from NERFdataset import dataset

import torch.nn as nn


import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch.nn as nn
from collections import defaultdict

import random

import wandb

from ddpm_pipeline import DDPMPipeline

D = 128

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
    
def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()
    
class NerfDiff(nn.Module):
    def __init__(self):
        super().__init__()

        self.ddpm_pipeline = DDPMPipeline(image_size=D)


def losses(model, data):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)

    loss_train_sd = model.module.ddpm_pipeline.train_step_uc(img[:,0])

    loss = loss_train_sd 

    loss_details = { 'sd': loss_train_sd}

    return loss, loss_details

def sample(model, data):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)

    samples = model.module.ddpm_pipeline.sample_uc((1,3,D,D))

    return samples




def train(rank, world_size, transfer="", use_wandb=False):

    setup(rank, world_size)
    
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
    num_epochs = 1810
    image_size = D
    batch_size = 64
    acc_steps = 1
    n_sample = 1000

    n_workers = 6
    epochs_plot_loss = 50
    epochs_plot_loss2 = 200

    if use_wandb and rank==0:
        wandb.config.update(
	        {
            "train_batch_size":batch_size,
            }
        )


        
    d = dataset('train', path='data/SRN/cars_train', imgsize=image_size, normalize_first_view=False)


    loader = prepare(rank, world_size, d, batch_size=batch_size)
#    loader = DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers, pin_memory=True)
    
    # Model setting
    model = NerfDiff().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    

    use_amp=False
    optimizer = AdamW([{'params':model.module.ddpm_pipeline.parameters(), 'lr':1e-4}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer == "":
        step = 0
        start_epoch = 0
    else:
        print('transfering from: ', transfer)
        
        # Mapped ckpt loading

        ckpt = torch.load(os.path.join(transfer, 'ddpm-latest.pt'))#, map_location=map_location)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optim'])
        step = ckpt['step']
        start_epoch = ckpt['epoch']+1

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1, eta_min=1e-8, last_epoch=-1)

    t00 = t0 = time.monotonic()
    # Training loop
    for e in range(start_epoch, num_epochs):

        print(f'starting epoch {e}')
        
        model.train()
        
        lt = time.time()

        # For each sample in the dataset
        pbar = tqdm(loader)

        epoch_loss = 0.0
        running_loss = 0.0
        running_loss_rgb = 0.0
        epoch_loss_details = defaultdict(float)

        for data in pbar:

            #with torch.cuda.amp.autocast(enabled=use_amp):
               
            # Forward and loss compute
            loss, loss_details = losses(model, data)



            loss.backward()
            
            if (step+1)%acc_steps==0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                #scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss
            for d in loss_details:
                epoch_loss_details[d] += loss_details[d].item()

            running_loss += loss.item()

            if step % n_sample == 0 and rank==0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_description(f'loss : {running_loss_rgb/n_sample} {running_loss/n_sample} {lr}')
                if use_wandb:
                    wandb.log(
                            {
                                "walltime": time.monotonic() - t00,
                                "train/step": step,
                                "train/epoch": e,
                                **{f"train/{k}": v for k, v in loss_details.items()},
                                #**{f"train/{k}": v for k, v in train_metrics.items()},
                            }
                        )

            
                running_loss = 0.0
                running_loss_rgb = 0.0

                formatted_images = []

                samples = sample(model, data)

                for i in range(len(samples)):
                    s = (255*(samples[i].clip(0,1)).cpu().detach().numpy().transpose(1,2,0)).astype(np.uint8)
                    imwrite(f'output/sample-{step:06d}-{i}.png', s)
                    formatted_images.append(wandb.Image(s, caption=f"sample-{i}"))
                if use_wandb:
                    wandb.log({"validation": formatted_images})

            step += 1
        scheduler.step()
            
        print('loss epoch', epoch_loss / len(loader))
        epoch_loss_details = {k:v/len(loader) for k,v in epoch_loss_details.items()}
        print('loss epoch', epoch_loss_details)

        if use_wandb and rank==0:
            wandb.log(
                    {

                                **{f"train/epoch-{k}": v for k, v in epoch_loss_details.items()},
                                #**{f"train/{k}": v for k, v in train_metrics.items()},
                    }
                    )

        # Epoch checkpoint save
        if (e+1) % epochs_plot_loss == 0 and rank==0:
            #pass
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, "output/ddpm-latest.pt")
            #torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"epoch-{e}.pt")
            
        if (e+1) % epochs_plot_loss2 == 0 and rank==0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"output/epoch-{e}.pt")

    

if __name__ == "__main__":
    MIN_GPUS = 1

    """"
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    args = parser.parse_args()
    """
    world_size = 1
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size
    )
   
