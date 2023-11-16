""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch.nn as nn 
import torch

def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    


class PatchEmbed_spa(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=13, patch_size=1, in_chans=145, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()   

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0]),  f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (W == self.img_size[1]), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PatchEmbed_chan(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=13, patch_size=1, in_chans=145, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = in_chans
        self.flatten = flatten

        self.proj = nn.Linear(self.grid_size[0] * self.grid_size[1], embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0]),  f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (W == self.img_size[1]), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        if self.flatten:
            x = x.flatten(2)  # BCHW -> BCN
        x = self.proj(x)
        x = self.norm(x)
        return x


class PositionEmbed(nn.Module):
    def __init__(self, num_patches=196, d_model=768, num_tokens=0):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.num_tokens = num_tokens
        assert self.num_tokens >=0, "num_tokens must be class token or no, so must 0 or 1"
        pe = torch.zeros(num_patches+self.num_tokens, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, num_patches + self.num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe = pe.cuda()
        self.register_buffer('pe', pe)

    def __call__(self):
        return self.pe


    


    