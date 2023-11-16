import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange

def build_mask_spa(mask_index, patch_size, img_size):
    num_pathces = img_size // patch_size
    mask_map = torch.zeros((img_size, img_size)).float()
    # reshape the h w -> n c 
    mask_map = rearrange(mask_map, '(h p1) (w p2) -> (h w) (p1 p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size)
    mask_index = [index-1 for index in mask_index ]
    mask_map[mask_index] = 1.
    # reshape the n c -> h w
    mask_map = rearrange(mask_map, '(h w) (p1 p2) -> (h p1) (w p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size)
    return mask_map

def build_mask_chan(mask_index, channel_num,patch_size):
    mask_map = torch.zeros((channel_num,1)).float()
    mask_index = [index-1 for index in mask_index ]
    mask_map[mask_index] = 1.

    return mask_map

class MSELoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
    def forward(self, pred, target, mask_map):

        pred = pred * mask_map.to(self.device)
        target = target * mask_map.to(self.device)
        loss = F.mse_loss(pred, target)
        return loss 
    
    
