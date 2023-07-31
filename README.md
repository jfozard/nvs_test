# (Almost) Generative Novel View Synthesis with 3D-Aware Diffusion Models

Unofficial, *partial* and *totally non-faithful* PyTorch Re-Implementation of [GeNVS](https://github.com/NVlabs/genvs).

## Changes:

- ~~Input network is a UNet, based on ResNet34, rather than the DeepLabV3+ network in the original paper.~~ Have now implemented something close to the DeepLabV3+ network with skip connections from the paper. Code presently in dlv3 branch.

- View-conditioned denoiser uses the k-diffusion implementation of EDM at https://github.com/crowsonkb/k-diffusion 

- Auxillary losses applied on volume rendered RGB images at novel views (first three channels of rendered latents).
  This probably impedes optimal performance later, but permits pre-training of just the input network and NeRF,
  independent of the denoising network. This RGB rendering is by necessity blurry (and cannot deal with ambiguity),
  but seems to train much faster than the diffusion model.

- Also (small) auxillary losses applied to the occupancy of the rendered views. This attempted to stop the NeRF filling
  the entire volume of the frustrum, but had little effect in practice.

- Image to NeRF model tends towards partial billboarding, with detail placed between the object and the camera. Attempted to correct this by additional loss penalizing differences in depth and opacity between the image where the source view is in the same position as the camera, and the image where the source view is in a different position. This didn't seem to help massively - the model just seemed to generate more background density.

- Only one or two (not three) views supplied for each batch, in order to train at batch size>1 on consumer hardware. (For later training up to three views supplied.)

- Increased noise level in diffusion model - seems to help the model whilst training give better predictions of x0, conditioned on the NeRF renderings, far more quickly than the default setting - but may hinder overall performance of method

- Stochastic sampling - for whatever reason (insufficient training? discrepency between training and sampling noise levels?) the deterministic samplers perform poorly on this dataset. Here using 250 steps of the Euler-like method from the EDM paper.

- Simplistic autoregressive sampling - conditioning on supplied image, up to 4 intermediate images, and the previously generated image. Greatly improves sampling output but still flickers a bit with current trained model.

## Training:

```
python train.py transfer=path_to_ckpt
```
Config file in config/config.yaml


## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download`cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

From https://github.com/a6o/3d-diffusion-pytorch , there is a pickle file that contains available view-png files per object. 


## Pre-trained Model Weights:

Model weights available at https://huggingface.co/JFoz/test_nvs (for both main and dlv3 branches).

Training procedure complex - original model generated by pretraining 64x64 diffusion model and
image to NeRF models, combining them and then further finetuning at resolutions 64 and 128.

Increased SNR levels (lonormal distribution with mean 1.0, standard deviation 1.4) were used for this (see k-configs/config_64_cars_noisy.json), and then subsequently dropped later in training. This is much higher noise levels
than in the original paper, but we want good predictions of the denoised image at high noise levels.

Noise levels dropped slightly (mean 0.5, standard deviation 1.4) for further fine-tuning. Still considerably higher than those in the EDM paper (mean -1.2 standard deviation 1.2). Scope for further experimentation - this task is different to standard diffusion
as the conditioning with an extra image is far more informative than a text prompt.

To train the dlv3 branch, the original UNet translating images to NeRF parameters was replaced by the DeepLabV3+ like
model and the whole model trained for another ~ 3days.

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

### Autoregressive sampling

This produces much better results than sampling from a single view. Strongly suggests this is required for decent levels of multi-view consistency. Still struggles a little bit with flickering, and consistency of details between the different sides of the vehicle. Unclear if this is due to insufficient training of the Image -> NeRF network, or a deficiency because pairs of features on opposite sides of the vehicle can never appear together in a single image. In the latter case, cross attention between views (3DiM, nerfdiff) may be a sensible addition to the denoising model.

First frame of each video is the conditioning image.

https://github.com/jfozard/nvs_test/assets/4390954/716969c4-8061-45c0-95dd-47ade62fb305

https://github.com/jfozard/nvs_test/assets/4390954/e9bae498-50ff-49f2-98e4-43930158a446

https://github.com/jfozard/nvs_test/assets/4390954/580a5ba1-8354-4f5d-a674-f4f08ca2a208

https://github.com/jfozard/nvs_test/assets/4390954/a28173b0-9469-449c-a1a2-3499038e5629

https://github.com/jfozard/nvs_test/assets/4390954/82b1a47d-571d-4976-aa2c-0892dbca2c13



## TODO

- Further training of model on a real multi-GPU system.
- Investigate inference strategy further - which images to retain in conditioning, and whether to resample views?
- Increase augmentation amount - current denoising model struggles with views which differ substantially from training set.
- Train on larger, more general dataset.
- Explore noise range schedules during training - start with fairly high noise levels and drop over time.
- Also explore LR schedule.
- Get a decent pixelNeRF to use as a starting point for training
- Similarly, obtain a decent k-diffusion model rather than train from scratch.

## Acknowledgements

K-diffusion from Katherine Crawson and others https://github.com/crowsonkb/k-diffusion

NeRF rendering using an old version of ashawkey's excellent https://github.com/ashawkey/stable-dreamfusion

Some data pipeline from https://github.com/a6o/3d-diffusion-pytorch and https://github.com/halixness/distributed-3d-diffusion-pytorch
