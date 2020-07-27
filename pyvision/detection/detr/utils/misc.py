import torch
import torch.nn as nn
import torch.nn.functional as F  

from typing import List, Optional

class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[torch.Tensor]):

        self.tensors = tensors
        self.mask = mask

    def to(self, device):

        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask =  None
        
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        return  self.tensors, self.mask
    
    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(inp_list):
    """List[List[int]] -> List[int]"""
    maxes = inp_list[0]
    for sublist  in inp_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes



def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Converts a list of tensor-images[3, H, W]  into nested tensor object for 
    model input
    """
    if tensor_list[0].ndim == 3:

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_size = [len(tensor_list)] + max_size
        b, c, h, w = batch_size
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_size, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)    
        
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("Images can have ndim == 3 but found ", tensor_list[0].ndim)

    return NestedTensor(tensor, mask)



