import math
import torch
import torch.nn as nn

from .misc import NestedTensor

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version  of the position embedding as defined
    in the attention is all you need paper, but is generalized to work on
    images 
    """
    def __init__(self, num_pos_feats=64, temp=10000, norm=False, scale=None):
        super().__init__()
        
        self.num_pos_feats = num_pos_feats
        self.temp = temp
        self.norm  = norm
    

        if scale is not None and norm is False:
            raise ValueError("normalize should be true if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        
        self.scale = scale
    
    def forward(self, tensor_list: NestedTensor):
        
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        assert mask is not None

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.norm:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temp ** (2 * (dim_t // 2) / self.num_pos_feats)

        x_pos = x_embed[:, :, :, None] / dim_t 
        y_pos = y_embed[:, :, :, None] / dim_t
        x_pos = torch.stack((x_pos[:,:,:,0::2].sin(), x_pos[:,:,:,1::2].cos()), dim=4).flatten(3)
        y_pos = torch.stack((y_pos[:,:,:,0::2].sin(), y_pos[:,:,:,1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((y_pos, x_pos), dim=3).permute(0, 3, 1, 2)
        
        return pos
