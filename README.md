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
![cars-conditioning-1-000000](https://github.com/jfozard/nvs_test/assets/4390954/0574042b-e372-4743-9433-d0cf209cd5a7)

Novel views generated (upper - denoised samples, lower- RGB renderings from NeRF)


Stochastic sampling

https://github.com/jfozard/nvs_test/assets/4390954/cd744bbd-9bdb-427a-a46c-be70f1a65e19

Deterministic sampling

https://github.com/jfozard/nvs_test/assets/4390954/fc45bd4d-0e34-455f-8d34-56ea20176c6b

Sampling progress (stochastic)

https://github.com/jfozard/nvs_test/assets/4390954/bdd2b8a1-8e31-4ee7-9949-6ca0efc9b98f

Sampling progress (deterministic)

https://github.com/jfozard/nvs_test/assets/4390954/c1a5acfd-2d34-4d55-8d14-95d71141a806




![cars-conditioning-1-000001](https://github.com/jfozard/nvs_test/assets/4390954/6882b749-dd47-486e-9377-8bb4c19da051)



https://github.com/jfozard/nvs_test/assets/4390954/a76015f1-2dc5-421c-8d17-405758c2cecd

https://github.com/jfozard/nvs_test/assets/4390954/e2c8f175-c4a3-465a-ae14-4a39a11ff506





https://github.com/jfozard/nvs_test/assets/4390954/5e8cd52f-aff9-4b36-8074-bdbee2d35eb6



https://github.com/jfozard/nvs_test/assets/4390954/8c0aecd0-1744-4a0c-bfab-8803265905b4





#### Unconditional samples (Supply pure noise conditioning image to diffusing model)

Stochastic

![uc-step-1-000000-250](https://github.com/jfozard/nvs_test/assets/4390954/590c4fd7-9b47-4390-bdfb-ef50446639e0)

https://github.com/jfozard/nvs_test/assets/4390954/6fedfdae-7ae3-4e95-bb00-6af90235b3e0

Deterministic

![uc_det-step-1-000000-250](https://github.com/jfozard/nvs_test/assets/4390954/6771b11d-23ff-48f4-8928-993900186c5e)

https://github.com/jfozard/nvs_test/assets/4390954/2aa18fa4-3317-4d29-b3e8-97bf3f337f4e



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
