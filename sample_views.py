
import os

from imageio import imwrite
import numpy as np
from tqdm import tqdm
import time
import argparse

from einops import rearrange
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from NERFdataset_k import dataset

from nerf.utils import render_multi_view
import torch.nn as nn

from genvs_model import NerfDiff

from fastai.vision.models import resnet34
from fastai_unet import DynamicUnet
import torch.nn as nn
from collections import defaultdict

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

# Different tasks  
#   1) just train RGB
#   2) import pre-trained diffusion model (unconditional)
#   3) train at 64x64
#   4) train at 128x128, importing 64x64 model

# Options - augment inputs, diffaugment on outputs
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    return sampler, dataloader
    
def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()



import torch.nn as nn

@torch.no_grad()
def sample(model, data, source_view_idx, sample_view_batch=2):
    # model - NerfDiff model
    # data - dataset batch 
    # source_view_idx - list of view indicies used to generate NeRF
    # sample_view_batch - number of novel views to generate at the same time
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
    

    imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
    img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) 

    B, V = img_tp.shape[:2]
    img_tp = img_tp.view(B*V,*img_tp.shape[2:])
    triplanes = model.module.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
    triplanes = triplanes.view(B, V, *triplanes.shape[1:])[:,source_view_idx].contiguous()

    # Downsample targets

    poses = data['poses'].cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k'].cuda()
    camera_d = data['camera_d'].cuda()

    cameras  = (camera_k, camera_d, poses[:,source_view_idx].contiguous())

    render_output_views = []
    render_output_depth = []
    render_output_opacities = []

    Q = 2
    np = poses.shape[1]

    for q in range(0, Q, 2):
    	  


        render_poses = poses[:,q:min(np, q+Q)]    

        first_view, d1, o1 = render_multi_view(model.module.nerf, render_poses, intrinsics[0], triplanes, cameras)
        print('first_view', first_view.min(), first_view.max())

        first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
        first_view = first_view.view(B, Q, *first_view.shape[1:])
        first_view_rgb = first_view[:, :, :3]

        
        d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
        d1 = d1.view(B, Q, *d1.shape[1:])
        o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
        o1 = o1.view(B, Q, *o1.shape[1:])

        render_output_depth.append(d1)
        render_output_opacities.append(o1)


        samples1 = model.module.ddpm_pipeline.sample(first_view.view(B*Q, *first_view.shape[2:]))
        render_output_views.append(samples1.cpu())

    render_output_views = torch.stack(render_output_views, dim=1)


    return render_output_views, render_output_depth, render_output_opacities




def sample(rank, world_size, transfer="", use_wandb = False):

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
    image_size = 128
    batch_size = 1



    d = dataset('train',  imgsize=image_size)
    
    sampler, loader = prepare(rank, world_size, d, batch_size=batch_size)

    # Model setting
    model = NerfDiff.cuda()
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], output_device=rank)

    use_amp=False

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer !="":
             # inner_model.proj_in.weight

        print('resume from: ', transfer)

        ckpt = torch.load(os.path.join(transfer, 'large-k-multi-latest.pt'))#, map_location=map_location)

        model.module.load_state_dict(ckpt['model'])


        del ckpt



    model.eval()
    
    pbar = tqdm(loader)

    for step, data in enumerate(pbar):
        if step==1:
            cleanup()
            return

        original_views, render_output_views, render_output_depth, render_output_opacities = sample(model, data, [0])

        for k in range(original_views.shape[1]):
            output = np.concatenate((np.concatenate((img1.cpu().detach().numpy()[k].transpose(1,2,0), v1.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d1,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o1,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1),
                                np.concatenate((img2.cpu().detach().numpy()[k].transpose(1,2,0), v2.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d2,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o2,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1)), axis=0)
        
        
            output = np.concatenate(((original_views[0,k].transpose(1,2,0), 
                                      render_output_views[0,k].transpose(1,2,0))))
            
            
            output = (255*np.clip(output,0,1)).astype(np.uint8)
            imwrite(f'output_view/sample-{step:06d}-{k}.png', output)

        del original_views, render_output_views, render_output_depth, render_output_opacities

            
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(sample, args=(world_size,args.transfer), nprocs=world_size, join=True)
