""" Vision Transformer (ViT) in PyTorch
"""
import sys 
import math
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.VIT.layers.patch_embd import PatchEmbed_spa,PatchEmbed_chan,PositionEmbed
from net.VIT.layers.mlp import Mlp,Mlp_wo_gate
from net.VIT.layers.drop import DropPath
from net.VIT.layers.weight_init import trunc_normal_
from net.VIT.utils.mask_embeeding import MaskEmbeeding, UnMaskEmbeeding_spa,UnMaskEmbeeding_chan


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Block_wo_gate(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_wo_gate(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer_spa=PatchEmbed_spa,embed_layer_chan=PatchEmbed_chan, pos_embed="cosine",
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, pool='mean',
                 classification=False, vit_type="encoder", mask_ratio=0.75, MAE=True,
                 args = None
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            pos_embed (nn.Module): position embeeding layer cosine or learnable parameters
            norm_layer: (nn.Module): normalization layer
            pool: 'mean' or 'cls' for classification
            classification: True or False 
            vit_type: "encoder" or "decoder" for MAE
            mask_ratio: a ratio for mask patch numbers
            MAE: Use MAE for trainig 
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  
        self.classification = classification 
        self.mask_ratio = mask_ratio 
        self.vit_type = vit_type 
        self.MAE = MAE 
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
    
        self.patch_embed_spa = embed_layer_spa(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_chan = embed_layer_chan(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_spa = self.patch_embed_spa.num_patches
        num_patches_chan = self.patch_embed_chan.num_patches
        self.cls_token_spa = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_chan = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if pos_embed == "cosine":
            self.pos_embed_spa = PositionEmbed(num_patches_spa, embed_dim, self.num_tokens)().to(self.device)
            self.pos_embed_chan = PositionEmbed(num_patches_chan, embed_dim, self.num_tokens)().to(self.device)
        else:
            self.pos_embed_spa = nn.Parameter(torch.zeros(1, num_patches_spa + self.num_tokens, embed_dim)).to(self.device)
            self.pos_embed_chan = nn.Parameter(torch.zeros(1, num_patches_chan + self.num_tokens, embed_dim)).to(self.device)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.vit_type == "decoder":
            self.unmask_embed_spa = UnMaskEmbeeding_spa(img_size,
                                           embed_dim,
                                           in_chans,
                                           patch_size,
                                           num_patches_spa,args
                                           ).to(self.device)
            self.unmask_embed_chan = UnMaskEmbeeding_chan(img_size,
                                                        embed_dim,
                                                        in_chans,
                                                        patch_size,
                                                        num_patches_chan,args
                                                        ).to(self.device)
        # for MAE dropout is not use
        if self.MAE:
            dpr = [0.0 for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        if self.vit_type == "decoder":
            self.blocks_spa = nn.Sequential(*[
                Block_wo_gate(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
            self.blocks_chan = nn.Sequential(*[
                Block_wo_gate(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
        else:
            self.blocks_spa = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
            self.blocks_chan = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        self.pool = pool
        
        if self.classification:
            self.class_head = nn.Sequential(
                nn.Linear(self.num_features*2, self.num_features),
                nn.Linear(self.num_features,self.num_classes)
            )
        
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def autoencoder(self, x):
        """encoder the no mask patch embeeding with position embeeding
        Returns:
            norm_embeeding: encoder embeeding 
            sample_index:   a list of token used for encoder
            mask_index      a list of token mask 
        """

        x_spa = self.patch_embed_spa(x)
        x_chan = self.patch_embed_chan(x)
        # add cls token for classification
        dummpy_token_spa = self.cls_token_spa.expand(x_spa.shape[0], -1, -1)
        dummpy_token_chan = self.cls_token_chan.expand(x_chan.shape[0], -1, -1)
        x_spa = torch.cat((dummpy_token_spa, x_spa), dim=1)
        x_chan = torch.cat((dummpy_token_chan, x_chan), dim=1)
        # print(self.pos_embed_spa.device)
        # print(x_spa.device)
        x_spa = x_spa + self.pos_embed_spa
        x_chan = x_chan+ self.pos_embed_chan

        # mask the patchemb&posemb
        mask_patch_embeeding_spa, sample_index_spa, mask_index_spa = MaskEmbeeding(x_spa, self.mask_ratio)
        mask_patch_embeeding_chan, sample_index_chan, mask_index_chan = MaskEmbeeding(x_chan, self.mask_ratio)

        x_spa = self.blocks_spa(mask_patch_embeeding_spa)
        x_chan = self.blocks_chan(mask_patch_embeeding_chan)
        norm_embeeding_spa = self.norm(x_spa)
        norm_embeeding_chan = self.norm(x_chan)

        return norm_embeeding_spa, sample_index_spa, mask_index_spa,norm_embeeding_chan, sample_index_chan, mask_index_chan
    
    
    def decoder(self, x_spa, sample_index_spa, mask_index_spa,x_chan, sample_index_chan, mask_index_chan):
        """decoder the all patch embeeding with the mask and position embeeding 
        """
        # unmask the patch embeeidng with the encoder embeeding 
        # print(x_spa.device)
        decoder_embed_spa = self.unmask_embed_spa(x_spa, sample_index_spa, mask_index_spa)
        decoder_embed_chan = self.unmask_embed_chan(x_chan, sample_index_chan, mask_index_chan)
        x_spa = decoder_embed_spa + self.pos_embed_spa
        x_chan = decoder_embed_chan + self.pos_embed_chan
        # decoder
        decoder_embeeding_spa = self.blocks_spa(x_spa)
        decoder_embeeding_chan = self.blocks_chan(x_chan)
        return decoder_embeeding_spa,decoder_embeeding_chan
    
    
    def forward_features(self, x):
        """Return the layernormalization features
        """
        x_spa = self.patch_embed_spa(x)
        x_chan = self.patch_embed_chan(x)
        # add cls token for classification
        cls_token_spa = self.cls_token_spa.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token_chan = self.cls_token_chan.expand(x.shape[0], -1, -1)
        x_spa = torch.cat((cls_token_spa, x_spa), dim=1)
        x_chan = torch.cat((cls_token_chan, x_chan), dim=1)
        
        x_spa = self.pos_drop(x_spa + self.pos_embed_spa)
        x_chan = self.pos_drop(x_chan + self.pos_embed_chan)
        x_spa = self.blocks_spa(x_spa)
        x_chan = self.blocks_chan(x_chan)
        x_spa = self.norm(x_spa)
        x_chan = self.norm(x_chan)
        return x_spa,x_chan

    # def forward(self, x):
    #     x_spa,x_chan = self.forward_features(x)
    #     # print(x_spa.shape)
    #     # print(x_chan.shape)
    #     if self.pool == "mean":
    #         x_spa = x_spa.mean(dim=1)
    #         x_chan = x_chan.mean(dim=1)
    #     elif self.pool == "cls":
    #         x_spa = x_spa[:, 0]  # cls token
    #         x_chan = x_chan[:, 0]
    #     else:
    #         raise ValueError("pool must be 'cls' or 'mean' ")
        
    #     assert x_spa.shape[1] == self.num_features, "outputs must be same with the features"
    #     if self.classification:
    #         # x = self.class_head(x_spa+x_chan)
    #         # print(x_spa.shape)
    #         x = torch.cat((x_spa,x_chan),dim=1)
    #         x = self.class_head(x)
    #     return x_spa,x_chan

    def forward(self, x):
        x_spa,x_chan = self.forward_features(x)

        if self.pool == "mean":
            x_spa = x_spa.mean(dim=2)
            x_chan = x_chan.mean(dim=2)
        elif self.pool == "cls":
            x_spa = x_spa[:, 0]  # cls token
            x_chan = x_chan[:, 0]
        else:
            raise ValueError("pool must be 'cls' or 'mean' ")
        
        # assert x_spa.shape[1] == self.num_features, "outputs must be same with the features"
        # if self.classification:
            # x = self.class_head(x_spa+x_chan)
            # print(x_spa.shape)
            # x = torch.cat((x_spa,x_chan),dim=1)
            # x = self.class_head(x)
        return x_spa,x_chan



