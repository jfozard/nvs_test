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

Sample weights at https://www.dropbox.com/s/3z0c1wgnw4o23en/large-k-multi-latest.pt?dl=0.

Model was pre-trained by training the projection UNet / NeRF with just supervision of RGB outputs (~12hr 3090).

A separate (unconditional) denoising model was trained (~12hr) on the cars dataset at 64x64 using the k-diffusion
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

![conditioning-000000](https://github.com/jfozard/nvs_test/assets/4390954/39d3d6fe-64d4-4b4b-8c3f-1ca4533c3801)

![conditioning-000001](https://github.com/jfozard/nvs_test/assets/4390954/8ffde116-57b0-4ef1-8384-befa17e9de61)

Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)

https://github.com/jfozard/nvs_test/assets/4390954/9f1839e1-cb54-4e56-874c-5b6ee7eb0eff

https://github.com/jfozard/nvs_test/assets/4390954/ad85cb01-93e7-44a4-b766-ef277a63d7c8

Unconditional samples (Supply noise conditioning image to diffusing model)

![uc-sample-000000-0](https://github.com/jfozard/nvs_test/assets/4390954/1d8de1dd-38f2-4201-b83c-be37df7d673d)

![uc-sample-000000-1](https://github.com/jfozard/nvs_test/assets/4390954/a6238dd8-d2a6-4638-a53d-4dafe062422d)

![uc-sample-000000-2](https://github.com/jfozard/nvs_test/assets/4390954/c351ee7c-d3b9-4b73-a4b8-52c6d6dafaef)

![uc-sample-000000-16](https://github.com/jfozard/nvs_test/assets/4390954/bd2d8957-3549-4bab-ad72-13bf96e9efb6)









## Acknowledgements

K-diffusion from Katherine Crawson and others https://github.com/crowsonkb/k-diffusion

Nerf rendering using an old version of ashawkey's excellent https://github.com/ashawkey/stable-dreamfusion

Some data pipeline from https://github.com/a6o/3d-diffusion-pytorch and https://github.com/halixness/distributed-3d-diffusion-pytorch
