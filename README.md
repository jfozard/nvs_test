# (Almost) Generative Novel View Synthesis with 3D-Aware Diffusion Models

Unofficial, *partial* and *totally non-faithful* PyTorch Re-Implementation of [GeNVS](https://github.com/NVlabs/genvs).

## Changes:

- Input network is a UNet, based on ResNet34, rather than the DeepLabV3+ network in the original paper.

- View-conditioned denoiser uses the k-diffusion implementation of EDM https://github.com/crowsonkb/k-diffusion
This network is considerably larger (500M params?) than the denoising network in the manuscript. Chosen by accident
owing to defaults in k-diffusion repo.

- Auxillary losses applied on volume rendered RGB images at novel views (first three channels of rendered latents).
  This probably impedes optimal performance later, but permits pre-training of just the input network and NeRF,
  independent of the denoising network. This RGB rendering is by necessity blurry (and cannot deal with ambiguity),
  but seems to train much faster than the diffusion model.

- Also (small) auxillary losses applied to the occupancy of the rendered views. This attempted to stop the NeRF filling
  the entire volume of the frustrum, but had little effect in practice.

- Only one or two (not three) views supplied for each batch, in order to train at batch size>1 on consumer hardware.

- Increased noise level in diffusion model - seems to help the model whilst training give better predictions of x0, conditioned on the NeRF renderings, far more quickly than the deault setting - but may hinder overall performance of method

## Training:

```
python train.py transfer=path_to_ckpt
```
Config file in config/config.yaml


## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download`cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

From https://github.com/a6o/3d-diffusion-pytorch , there is a pickle file that contains available view-png files per object. 


## Pre-trained Model Weights:

Sample weights at X.

Model was pre-trained by training the projection UNet / NeRF with just supervision of RGB outputs (~12hr 3090).
A seperate (unconditional) denoising model was trained (~12hr) on the cars dataset at 64x64 using the k-diffusion
repository. (Checkpoint at X).
These two sets of model weights were then combined, and finetuned (at output resolution of 64x64). (~24hr 3090)
Increased SNR levels (lonormal distribution with mean 1.0, standard deviation 1.4) were used for this (see k-configs/config_64_cars_noisy.json).
This seems sensible as unlike standard text-to-image diffusion models, the additional information
from the rendered view should permit reasonably good prediction of the densoised image x_0 even at high noise levels.
The standard schedule concentrates mostly on lower ones, at which noise levels features such as the colour
of the car can be extracted easily from the noisy image, discarding additional cues from the rendered image.
Training was then scaled up to 128x128, extending the denoising unet with extra convolutional layers.
After another bit of training (8-12 hr), noise levels were dropped slightly.



## Current results

```
python sample_views.py --transfer .
```

These clearly need a bit more training!

Conditioning images


Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)








## Acknowledgements

K-diffusion from Katherine Crawson and others https://github.com/crowsonkb/k-diffusion
Nerf rendering using an old version of ashawkey's excellent https://github.com/ashawkey/stable-dreamfusion
Some data pipeline from https://github.com/a6o/3d-diffusion-pytorch and https://github.com/halixness/distributed-3d-diffusion-pytorch
