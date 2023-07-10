# (Almost) Generative Novel View Synthesis with 3D-Aware Diffusion Models

Unofficial, *partial* and *totally non-faithful* PyTorch Re-Implementation of [GeNVS](https://github.com/NVlabs/genvs).

## Changes:

- Input network is a UNet, based on ResNet34, rather than the DeepLabV3+ network in the original paper.

- View-conditioned denoiser uses the k-diffusion implementation of EDM https://github.com/crowsonkb/k-diffusion
  This network is a bit larger (110M params?) than the denoising network in the manuscript (90M). Chosen by accident
  owing to defaults in k-diffusion repo.

- Auxillary losses applied on volume rendered RGB images at novel views (first three channels of rendered latents).
  This probably impedes optimal performance later, but permits pre-training of just the input network and NeRF,
  independent of the denoising network. This RGB rendering is by necessity blurry (and cannot deal with ambiguity),
  but seems to train much faster than the diffusion model.

- Also (small) auxillary losses applied to the occupancy of the rendered views. This attempted to stop the NeRF filling
  the entire volume of the frustrum, but had little effect in practice.

- Only one or two (not three) views supplied for each batch, in order to train at batch size>1 on consumer hardware. (For later training up to three views supplied.)

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

Sample weights at 

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

Supplemented by additional ~24hr training on H100 at batch size 8, accumulation 8, and up to 3 input views.
(See https://wandb.ai/john-fozard/genvs/runs/fwxxsve2 )

Loss curve suggests learning rate schedule might need tuning.



## Current results

```
python sample_views.py --transfer .
```

These clearly need a bit more training!

#### Conditioned on a single view

Conditioning image

![conditioning-000000](https://github.com/jfozard/nvs_test/assets/4390954/bdc0d595-aae6-43f3-8acb-ac79d880161a)

Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)

https://github.com/jfozard/nvs_test/assets/4390954/e8cab1ae-a7ae-4bd0-b11d-8a103ed5fdfb


![conditioning-000001](https://github.com/jfozard/nvs_test/assets/4390954/4f04e2fb-3ea0-4d62-a637-5b600f0552da)

https://github.com/jfozard/nvs_test/assets/4390954/c8a61989-df03-4ddc-b950-4d1bbd13a764



#### Conditioned on two views


Conditioning images

![conditioning-2-000000](https://github.com/jfozard/nvs_test/assets/4390954/2bad2f1b-0c3f-4621-bae7-a99a086efee6)


Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)


https://github.com/jfozard/nvs_test/assets/4390954/be3d3a38-f46b-46d1-8c0d-0b74702dca82



https://github.com/jfozard/nvs_test/assets/4390954/0ebdc706-9f74-4ec2-a63e-194365ba1787

![conditioning-2-000002](https://github.com/jfozard/nvs_test/assets/4390954/22e8f0d7-925f-4c40-94d9-555ec129227d)





#### Unconditional samples (Supply pure noise conditioning image to diffusing model)

![uc-sample-000000-0](https://github.com/jfozard/nvs_test/assets/4390954/67825263-c5f8-44ab-a532-b1d02f79e9b5)

![uc-sample-000000-1](https://github.com/jfozard/nvs_test/assets/4390954/310c63fa-0304-48d5-bca9-7e46102a1342)

![uc-sample-000000-2](https://github.com/jfozard/nvs_test/assets/4390954/e441616c-8430-475d-a635-312df65f816e)

![uc-sample-000000-3](https://github.com/jfozard/nvs_test/assets/4390954/7f956fc2-0da8-4010-948c-dae115afeff3)



## TODO

- Further training of model on a real multi-GPU system.
- Investigate inference strategy further - which images to retain in conditioning, and whether to resample views?
- Implement DeepLabV3+ (with batchnorm and dropout removed from torchvision network).
- Increase augmentation amount - current denoising model struggles views which differ substantially from training set.
- Train on larger, more general dataset.
- Explore noise range schedules during training - start with fairly high noise levels and drop over time.
- Also explore LR schedule.






## Acknowledgements

K-diffusion from Katherine Crawson and others https://github.com/crowsonkb/k-diffusion

Nerf rendering using an old version of ashawkey's excellent https://github.com/ashawkey/stable-dreamfusion

Some data pipeline from https://github.com/a6o/3d-diffusion-pytorch and https://github.com/halixness/distributed-3d-diffusion-pytorch
