
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
import torch.nn.functional as F
from NERFdataset_k import dataset
from nerf.network_multi import NeRFNetwork
from nerf.utils import render_multi_view
import torch.nn as nn

from fastai.vision.models import resnet34
from fastai_unet import DynamicUnet
import torch.nn as nn
from collections import defaultdict

import random

import wandb

from k_pipeline import KPipeline
#from sd_pipeline import SDPipeline

def make_input_model(size=128, out_dim=3*48):
    m = resnet34()
    m = nn.Sequential(*list(m.children())[:-2])
    # Patch input layer
    #m[0] = nn.Conv2d(3+16, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = DynamicUnet(m, out_dim, (size, size), norm_type=None)
    return model



class NerfDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_unet = make_input_model(size=128, out_dim=16*64)
#        self.triplanes = nn.Parameter(torch.rand(1, 48*3, 128, 128))
        self.nerf = NeRFNetwork(color_feat_dim=16)
        self.ddpm_pipeline = KPipeline('configs/config_upscale_noisy.json')


def losses(model, data):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)

    
    B, V = targets.shape[:2]


    #targets= F.interpolate(targets.view(B*V, *targets.shape[2:]), scale_factor = 0.5).view(B, V,3, 64, 64)

    # Downsample targets
 
    nv = random.randint(1,2)

    imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
    img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) # 0 for orig xkpt

    B, V = img_tp.shape[:2]
    img_tp = img_tp.view(B*V,*img_tp.shape[2:])
    triplanes = model.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
    triplanes = triplanes.view(B, V, *triplanes.shape[1:])[:,:nv].contiguous()

    # Downsample targets


    poses = data['poses'].cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k'].cuda()
    camera_d = data['camera_d'].cuda()
    #print('intrinsics', intrinsics.shape)


    cameras  = (camera_k, camera_d, poses[:,:nv])
    #print('camera_poses', cameras[-1].shape)

    Q = 2

    render_poses = torch.cat([poses[:,:(Q-1)], poses[:,-1:]], dim=1)

    first_view, d1, o1 = render_multi_view(model.nerf, render_poses, intrinsics[0], triplanes, cameras)
    
    #print('first_view', first_view.shape)
    #print(first_view.view(B*Q, *first_view.shape[2:]).shape)
    first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
    first_view = first_view.view(B, Q, *first_view.shape[1:])
    first_view_rgb = first_view[:, :, :3]

    d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    d1 = d1.view(B, Q, *d1.shape[1:])
    o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    o1 = o1.view(B, Q, *o1.shape[1:])
    """
    d1 = d1.expand(-1,-1,3,-1,-1)
    o1 = o1.expand(-1,-1,3,-1,-1)
    """
    #print(first_view_rgb.shape, targets.shape)
    loss_rgb = 0.01*((first_view_rgb[:,0] - targets[:,0:1])**2).mean() + 1.0*((first_view_rgb[:,-1] - targets[:,-1:])**2).mean() 
    loss_opacity = 0.001*(o1.mean())


    # Denoise

    bsz = img.shape[0]
    cond_flag = 0 if random.random()>0.1 else 1
    if random.random() > 0.2:
        loss_train_sd = model.ddpm_pipeline.train_step(targets[:,-1], first_view[:,-1], cond_flag=cond_flag)
    else:
        loss_train_sd = model.ddpm_pipeline.train_step(targets[:,0], first_view[:,0], cond_flag=cond_flag) # ! Target was wrong before !


    loss = loss_train_sd +loss_rgb + loss_opacity #+ loss_train_sd
    #print(loss_rgb.item(), loss_opacity.item(), loss_train_sd.item())

    loss_details = { 'rgb': loss_rgb, 'opacity': loss_opacity, 'sd': loss_train_sd}

    return loss, loss_details

@torch.no_grad()
def sample(model, data, nv):
    img = data['imgs'].cuda()
    targets = 0.5*(img+1)
    
    B, V = targets.shape[:2]


    #targets= F.interpolate(targets.view(B*V, *targets.shape[2:]), scale_factor = 0.5).view(B, V, 3, 64, 64)


    imagenet_stats = (torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
    img_tp = (targets - (imagenet_stats[0][None,None,:,None,None]))/(imagenet_stats[1][None,None,:,None,None]) # 0 for orig xkpt

    B, V = img_tp.shape[:2]
    img_tp = img_tp.view(B*V,*img_tp.shape[2:])
    triplanes = model.input_unet(torch.flip(img_tp[:,:],[2])) # Triplanes for the first view
    triplanes = triplanes.view(B, V, *triplanes.shape[1:])[:,:nv].contiguous()

    # Downsample targets


    poses = data['poses'].cuda()
    intrinsics = data['intrinsics']
    camera_k = data['camera_k'].cuda()
    camera_d = data['camera_d'].cuda()
    #print('intrinsics', intrinsics.shape)


    cameras  = (camera_k, camera_d, poses[:,:nv])
    #print('camera_poses', cameras[-1].shape)

    Q = 2

    #render_poses = torch.cat([poses[:,:(Q-1)], poses[:,-1:]], dim=1)
    render_poses = poses[:,:(Q)]    

#    first_view, d1, o1 = render_multi_view(model.nerf, poses[:,:Q], intrinsics[0], triplanes, cameras)
    first_view, d1, o1 = render_multi_view(model.nerf, render_poses, intrinsics[0], triplanes, cameras)
    print('first_view', first_view.min(), first_view.max())
    #print(first_view.view(B*Q, *first_view.shape[2:]).shape)
    first_view= F.interpolate(first_view.view(B*Q, *first_view.shape[2:]), scale_factor = 2)
    first_view = first_view.view(B, Q, *first_view.shape[1:])
    first_view_rgb = first_view[:, :, :3]

    #print('d1', d1.shape)

    
    d1 = F.interpolate(d1.view(B*Q,*d1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    d1 = d1.view(B, Q, *d1.shape[1:])
    o1 = F.interpolate(o1.view(B*Q,*o1.shape[2:]), scale_factor = 2).expand(-1,3,-1,-1)
    o1 = o1.view(B, Q, *o1.shape[1:])
    """
    d1 = d1.expand(-1,-1,3,-1,-1)
    o1 = o1.expand(-1,-1,3,-1,-1)    
    """
    # Denoise

    #print('d1', d1.shape)

    samples1 = model.ddpm_pipeline.sample(first_view.view(B*Q, *first_view.shape[2:]))
    #samples2 = model.ddpm_pipeline.sample(second_view[:,0])


    return targets[:,0], first_view_rgb[:,0], d1[:,0], o1[:,0], targets[:,1], first_view_rgb[:,1], d1[:,1], o1[:,1], samples1 #torch.cat((samples1, samples2), dim=0)




def train(rank, world_size, transfer=""):

    use_wandb = False

    if use_wandb:
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
    image_size = 128
    batch_size = 2
    acc_steps = 16
    n_sample = 1000

    n_workers = 6
    epochs_plot_loss = 5
    epochs_plot_loss2 = 50

    if use_wandb:
        wandb.config.update(
	        {
            "train_batch_size":batch_size,
            }
        )

    d = dataset('train',  imgsize=image_size)

    loader = DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
    
    # Model setting
    model = NerfDiff().to('cuda:0')

    print([m.shape for m in model.parameters()])

    use_amp=False
    optimizer = AdamW([{'params':model.input_unet.parameters(), 'lr':1e-4}, {'params':model.nerf.parameters(), 'lr':2e-4}, {'params':model.ddpm_pipeline.parameters(), 'lr':2e-4}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    #optimizer = AdamW([{'params':model.ddpm_pipeline.parameters(), 'lr':2e-4}], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2) # NERF
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load saved model if defined
    if transfer == "":
        step = 0
        start_epoch = 0
    elif False: #transfer !="":
             # inner_model.proj_in.weight

        print('resume from: ', transfer)


        ckpt = torch.load(os.path.join(transfer, 'k-multi-latest.pt'))#, map_location=map_location)

        new_ckpt = {}
        for k, v in ckpt['model'].items():
            if 'd_blocks' not in k:
                new_ckpt[k] = v
            else:
                u = k.split('.')
                j = u.index('d_blocks')+1
                idx = k.split('.')[j]
                new_k = '.'.join(u[:j])+'.' +str(int(idx)+1) + '.' + '.'.join(u[j+1:])
                new_ckpt[new_k] = v
 
        

        model.load_state_dict(new_ckpt, strict=False)
             
  #      with torch.no_grad():
 #           print(list(ckpt['model']))
#
        #    quit()
        step = ckpt['step']
        start_epoch = ckpt['epoch']+1   
    else:
        print('transfering from: ', transfer)
        
        # Mapped ckpt loading


        
        # inner_model.proj_in.weight

   

        ckpt = torch.load(os.path.join(transfer, 'large-k-multi-latest.pt'))#, map_location=map_location)
        with torch.no_grad():
            m = ckpt['model']#
            print('ddpm', list(m.keys())[:10])
            model.load_state_dict(ckpt['model'], strict=False)
             
      

            """
            m = ckpt['model']#
            print('copy', list(m.keys())[:10])
            key = 'inner_model.proj_in.weight'
        
            m[key] = torch.cat((m[key], torch.zeros((128,16,1,1), dtype=m[key].dtype, device=m[key].device )), dim=1)
            m[key].requires_grad_(True)
            print(key, m[key].shape)
            #m['inner_model.proj_in.bias'].requires_grad_(True)
            model.ddpm_pipeline.inner_model.load_state_dict(m)#, strict=False)
            """

        step = ckpt['step']
        start_epoch = ckpt['epoch']+1

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1, eta_min=1e-8, last_epoch=-1)

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

            with torch.cuda.amp.autocast(enabled=use_amp):
               
            # Forward and loss compute
                loss, loss_details = losses(model, data)



            scaler.scale(loss).backward()
            
            if (step+1)%acc_steps==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                #scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss
            for d in loss_details:
                epoch_loss_details[d] += loss_details[d].item()

            running_loss += loss.item()
            #running_loss_rgb += loss_details['rgb'].item()



            if step % n_sample == 0:
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
                for vv in [1,2]:
                    img1, v1, d1, o1, img2, v2, d2, o2, samples = sample(model, data, vv)
                    #targets, samples = sample(model, data)


                    print('img1', img1.shape)
                    for k in range(len(img1)):
                        output = np.concatenate((np.concatenate((img1.cpu().detach().numpy()[k].transpose(1,2,0), v1.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d1,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o1,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1),
                                            np.concatenate((img2.cpu().detach().numpy()[k].transpose(1,2,0), v2.cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(d2,0,1).cpu().detach().numpy()[k].transpose(1,2,0), torch.clip(o2,0,1).cpu().detach().numpy()[k].transpose(1,2,0)), axis=1)), axis=0)
                    

                        output = (255*np.clip(output,0,1)).astype(np.uint8)
                        imwrite(f'output3/full-{step:06d}-{vv}-{k}.png', output)
                        if use_wandb:
                            formatted_images.append(wandb.Image(output, caption=f"{vv}-{k}"))

                    for i in range(len(samples)):
                        s = (255*(samples[i].clip(0,1)).cpu().detach().numpy().transpose(1,2,0)).astype(np.uint8)
                        imwrite(f'output3/sample-{step:06d}-{vv}-{i}.png', s)
                        if use_wandb:
                            formatted_images.append(wandb.Image(s, caption=f"sample-{vv}-{i}"))


                del img1, v1, d1, o1, img2, v2, d2, o2, samples
                #output = np.concatenate(((255*targets.cpu().detach().numpy()[0].transpose(1,2,0)).astype(np.uint8), 255*samples.cpu().detach().numpy()[0].transpose(1,2,0)), axis=1)


                #imwrite(f'copy-{step:06d}.png', (output).astype(np.uint8))


                if use_wandb:
                    wandb.log({"validation": formatted_images})

            step += 1
        scheduler.step()
            
        print('loss epoch', epoch_loss / len(loader))
        epoch_loss_details = {k:v/len(loader) for k,v in epoch_loss_details.items()}
        print('loss epoch', epoch_loss_details)

        if use_wandb:
            wandb.log(
                 {

                            **{f"train/epoch-{k}": v for k, v in epoch_loss_details.items()},
                            #**{f"train/{k}": v for k, v in train_metrics.items()},
            	  }
                )

        # Epoch checkpoint save
        if e % epochs_plot_loss == 0:
            #pass
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, "output3/large-k-multi-latest.pt")
            #torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"epoch-{e}.pt")
        if e % epochs_plot_loss2 == 0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, f"output/large-k-multi-epoch-{e}.pt")

    

if __name__ == "__main__":
    MIN_GPUS = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    args = parser.parse_args()

   
    train(0, 0, args.transfer)    
