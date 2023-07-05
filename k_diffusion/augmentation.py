from functools import reduce
import math
import operator

import numpy as np
from skimage import transform
import torch
from torch import nn

from kornia.geometry.transform import warp_affine

def translate2d(tx, ty):
    mat = [[1, 0, tx],
           [0, 1, ty],
           [0, 0,  1]]
    return torch.tensor(mat, dtype=torch.float32)


def scale2d(sx, sy):
    mat = [[sx,  0, 0],
           [ 0, sy, 0],
           [ 0,  0, 1]]
    return torch.tensor(mat, dtype=torch.float32)


def rotate2d(theta):
    mat = [[torch.cos(theta), torch.sin(-theta), 0],
           [torch.sin(theta),  torch.cos(theta), 0],
           [               0,                 0, 1]]
    return torch.tensor(mat, dtype=torch.float32)
    
    
def translate2d_b(tx_b, ty_b):
    mat = [[[1, 0, tx],
           [0, 1, ty],
           [0, 0,  1]] for tx, ty in zip(tx_b, ty_b)]
    return torch.tensor(mat, dtype=torch.float32)


def scale2d_b(sx_b, sy_b):
    mat = [[[sx,  0, 0],
           [ 0, sy, 0],
           [ 0,  0, 1]] for sx, sy in zip(sx_b, sy_b)]
    return torch.tensor(mat, dtype=torch.float32)


def rotate2d_b(theta_b):
    mat = [[[torch.cos(theta), torch.sin(-theta), 0],
           [torch.sin(theta),  torch.cos(theta), 0],
           [               0,                 0, 1]] for theta in theta_b]
    return torch.tensor(mat, dtype=torch.float32)


class KarrasAugmentationPipeline:
    def __init__(self, a_prob=0.12, a_scale=2**0.2, a_aniso=2**0.2, a_trans=1/8):
        self.a_prob = a_prob
        self.a_scale = a_scale
        self.a_aniso = a_aniso
        self.a_trans = a_trans

    def __call__(self, image):
        h, w = image.size
        mats = [translate2d(h / 2 - 0.5, w / 2 - 0.5)]

        # x-flip
        a0 = torch.randint(2, []).float()
        mats.append(scale2d(1 - 2 * a0, 1))
        # y-flip
        do = (torch.rand([]) < self.a_prob).float()
        a1 = torch.randint(2, []).float() * do
        mats.append(scale2d(1, 1 - 2 * a1))
        # scaling
        do = (torch.rand([]) < self.a_prob).float()
        a2 = torch.randn([]) * do
        mats.append(scale2d(self.a_scale ** a2, self.a_scale ** a2))
        # rotation
        do = (torch.rand([]) < self.a_prob).float()
        a3 = (torch.rand([]) * 2 * math.pi - math.pi) * do
        mats.append(rotate2d(-a3))
        # anisotropy
        do = (torch.rand([]) < self.a_prob).float()
        a4 = (torch.rand([]) * 2 * math.pi - math.pi) * do
        a5 = torch.randn([]) * do
        mats.append(rotate2d(a4))
        mats.append(scale2d(self.a_aniso ** a5, self.a_aniso ** -a5))
        mats.append(rotate2d(-a4))
        # translation
        do = (torch.rand([]) < self.a_prob).float()
        a6 = torch.randn([]) * do
        a7 = torch.randn([]) * do
        mats.append(translate2d(self.a_trans * w * a6, self.a_trans * h * a7))

        # form the transformation matrix and conditioning vector
        mats.append(translate2d(-h / 2 + 0.5, -w / 2 + 0.5))
        mat = reduce(operator.matmul, mats)
        cond = torch.stack([a0, a1, a2, a3.cos() - 1, a3.sin(), a5 * a4.cos(), a5 * a4.sin(), a6, a7])
	
        # apply the transformation
        image_orig = np.array(image, dtype=np.float32) / 255
        if image_orig.ndim == 2:
            image_orig = image_orig[..., None]
        tf = transform.AffineTransform(mat.numpy())
        image = transform.warp(image_orig, tf.inverse, order=3, mode='reflect', cval=0.5, clip=False, preserve_range=True)
        image_orig = torch.as_tensor(image_orig).movedim(2, 0) * 2 - 1
        image = torch.as_tensor(image).movedim(2, 0) * 2 - 1
        return image, image_orig, cond

    def get_mat_cond(self, image):
        h, w,_ = image.shape
        mats = [translate2d(h / 2 - 0.5, w / 2 - 0.5)]

        # x-flip
        a0 = torch.randint(2, []).float()
        mats.append(scale2d(1 - 2 * a0, 1))
        # y-flip
        do = (torch.rand([]) < self.a_prob).float()
        a1 = torch.randint(2, []).float() * do
        mats.append(scale2d(1, 1 - 2 * a1))
        # scaling
        do = (torch.rand([]) < self.a_prob).float()
        a2 = torch.randn([]) * do
        mats.append(scale2d(self.a_scale ** a2, self.a_scale ** a2))
        # rotation
        do = (torch.rand([]) < self.a_prob).float()
        a3 = (torch.rand([]) * 2 * math.pi - math.pi) * do
        mats.append(rotate2d(-a3))
        # anisotropy
        do = (torch.rand([]) < self.a_prob).float()
        a4 = (torch.rand([]) * 2 * math.pi - math.pi) * do
        a5 = torch.randn([]) * do
        mats.append(rotate2d(a4))
        mats.append(scale2d(self.a_aniso ** a5, self.a_aniso ** -a5))
        mats.append(rotate2d(-a4))
        # translation
        do = (torch.rand([]) < self.a_prob).float()
        a6 = torch.randn([]) * do
        a7 = torch.randn([]) * do
        mats.append(translate2d(self.a_trans * w * a6, self.a_trans * h * a7))

        # form the transformation matrix and conditioning vector
        mats.append(translate2d(-h / 2 + 0.5, -w / 2 + 0.5))
        mat = reduce(operator.matmul, mats)
        cond = torch.stack([a0, a1, a2, a3.cos() - 1, a3.sin(), a5 * a4.cos(), a5 * a4.sin(), a6, a7])
	
        return mat, cond
	
    def apply(self, image, mat, cond):
        # apply the transformation
        image_orig = np.array(image, dtype=np.float32) / 255
        if image_orig.ndim == 2:
            image_orig = image_orig[..., None]


        tf = transform.AffineTransform(mat.numpy())
        

        
#        image = transform.warp(image_orig, tf.inverse, order=3, mode='reflect', cval=0.5, clip=False, preserve_range=True)
        image = transform.warp(image_orig, tf.inverse, order=3, mode='reflect', cval=0.5, clip=False, preserve_range=True)
        image_orig = torch.as_tensor(image_orig).movedim(2, 0) * 2 - 1
        image = torch.as_tensor(image).movedim(2, 0) * 2 - 1
        return image, image_orig, cond


class KarrasDiffAugmentationPipeline:
    def __init__(self, a_prob=0.12, a_scale=2**0.2, a_aniso=2**0.2, a_trans=1/8):
        self.a_prob = a_prob
        self.a_scale = a_scale
        self.a_aniso = a_aniso
        self.a_trans = a_trans

    def __call__(self, image):
        b, c, h, w = image.shape
        
        h_b = torch.tensor([h]*b)
        w_b = torch.tensor([w]*b)
        mats = [translate2d_b(h_b / 2 - 0.5, w_b / 2 - 0.5)]

        # x-flip
        a0 = torch.randint(2, [b]).float()
        mats.append(scale2d_b(1 - 2 * a0, [1]))
        # y-flip
        do = (torch.rand([b]) < self.a_prob).float()
        a1 = torch.randint(2, [b]).float() * do
        mats.append(scale2d_b([1], 1 - 2 * a1))
        # scaling
        do = (torch.rand([b]) < self.a_prob).float()
        a2 = torch.randn([b]) * do
        mats.append(scale2d_b(self.a_scale ** a2, self.a_scale ** a2))
        # rotation
        do = (torch.rand([b]) < self.a_prob).float()
        a3 = (torch.rand([b]) * 2 * math.pi - math.pi) * do
        mats.append(rotate2d_b(-a3))
        # anisotropy
        do = (torch.rand([b]) < self.a_prob).float()
        a4 = (torch.rand([b]) * 2 * math.pi - math.pi) * do
        a5 = torch.randn([b]) * do
        mats.append(rotate2d_b(a4))
        mats.append(scale2d(self.a_aniso ** a5, self.a_aniso ** -a5))
        mats.append(rotate2d_b(-a4))
        # translation
        do = (torch.rand([b]) < self.a_prob).float()
        a6 = torch.randn([b]) * do
        a7 = torch.randn([b]) * do
        mats.append(translate2d_b(self.a_trans * w * a6, self.a_trans * h * a7))

        # form the transformation matrix and conditioning vector
        mats.append(translate2d_b(-h_b / 2 + 0.5, -w_b / 2 + 0.5))
        mat = reduce(operator.matmul, mats)
        cond = torch.stack([a0, a1, a2, a3.cos() - 1, a3.sin(), a5 * a4.cos(), a5 * a4.sin(), a6, a7], dim=1)

        return self.apply(image, mat, cond)

        
        
    def get_mat_cond(self, image):
        b, c, h, w = image.shape
        
        h_b = torch.tensor([h]*b)
        w_b = torch.tensor([w]*b)
        mats = [translate2d_b(h_b / 2 - 0.5, w_b / 2 - 0.5)]

        # x-flip
        a0 = torch.randint(2, [b]).float()
        mats.append(scale2d_b(1 - 2 * a0, [1]))
        # y-flip
        do = (torch.rand([b]) < self.a_prob).float()
        a1 = torch.randint(2, [b]).float() * do
        mats.append(scale2d_b([1], 1 - 2 * a1))
        # scaling
        do = (torch.rand([b]) < self.a_prob).float()
        a2 = torch.randn([b]) * do
        mats.append(scale2d_b(self.a_scale ** a2, self.a_scale ** a2))
        # rotation
        do = (torch.rand([b]) < self.a_prob).float()
        a3 = (torch.rand([b]) * 2 * math.pi - math.pi) * do
        mats.append(rotate2d_b(-a3))
        # anisotropy
        do = (torch.rand([b]) < self.a_prob).float()
        a4 = (torch.rand([b]) * 2 * math.pi - math.pi) * do
        a5 = torch.randn([b]) * do
        mats.append(rotate2d_b(a4))
        mats.append(scale2d_b(self.a_aniso ** a5, self.a_aniso ** -a5))
        mats.append(rotate2d_b(-a4))
        # translation
        do = (torch.rand([b]) < self.a_prob).float()
        a6 = torch.randn([b]) * do
        a7 = torch.randn([b]) * do
        mats.append(translate2d_b(self.a_trans * w * a6, self.a_trans * h * a7))

        # form the transformation matrix and conditioning vector
        mats.append(translate2d_b(-h_b / 2 + 0.5, -w_b / 2 + 0.5))
        mat = reduce(operator.matmul, mats)
        cond = torch.stack([a0, a1, a2, a3.cos() - 1, a3.sin(), a5 * a4.cos(), a5 * a4.sin(), a6, a7], dim=1)
        return mat.to(image.device), cond.to(image.device)

    def apply(self, image, mat, cond):
        # apply the transformation
        
        b, c, h, w = image.shape

        image_orig = image


        
        image = warp_affine(image, mat[:,:2,:].to(image.device), dsize=(h, w), mode='bilinear', padding_mode='reflection')


        return image, image_orig, cond
        
    def apply_numpy(self, image, mat, cond):
        h, w = image.shape[:2]
        # apply the transformation
        image_orig = np.array(image, dtype=np.float32) / 255
        if image_orig.ndim == 2:
            image_orig = image_orig[..., None]
        image_orig = torch.tensor(image_orig)
        image_orig = (image_orig.movedim(2,0)*2-1).unsqueeze(0)
        mat = torch.tensor(mat).unsqueeze(0)

        
        image = warp_affine(image_orig, mat[:,:2,:], dsize=(h, w), mode='bilinear', padding_mode='reflection')
        return image[0], image_orig[0], cond            

class KarrasAugmentWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    
    def forward(self, input, sigma, aug_cond=None, mapping_cond=None, **kwargs):
        if aug_cond is None:
            aug_cond = input.new_zeros([input.shape[0], 9])
        if mapping_cond is None:
            mapping_cond = aug_cond
        else:
            mapping_cond = torch.cat([aug_cond, mapping_cond], dim=1)
        return self.inner_model(input, sigma, mapping_cond=mapping_cond, **kwargs)

    def set_skip_stages(self, skip_stages):
        return self.inner_model.set_skip_stages(skip_stages)

    def set_patch_size(self, patch_size):
        return self.inner_model.set_patch_size(patch_size)