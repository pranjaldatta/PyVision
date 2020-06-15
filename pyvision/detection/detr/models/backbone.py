"""
The backbone modules are defined here 
"""

from typing import List, Dict

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from ..utils.misc import NestedTensor

class FrozenBatchNorm2d(nn.Module):
    """
    Custom batch norm layers where the batch stats and affine parameters
    are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without
    which any other models other than resnet[18, 24, 50, 101] produce nans 
    """
    def __init__(self, size):
        super(FrozenBatchNorm2d, self).__init__()
        
        self.register_buffer("weight", torch.ones(size))
        self.register_buffer("bias", torch.zeros(size))
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_var", torch.ones(size))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,strict,
                             missing_keys, unexpected_keys, error_msgs):
        
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata,strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone:nn.Module, train_backbone: bool, num_channels: int,
                return_interim_layers: bool):

        super().__init__()
        
        for name, param in backbone.named_parameters():
            if not train_backbone or "layer_2" not in name or "layer_3" not in name or "layer_4" not in name:
                param.requires_grad_(False)
        
        if return_interim_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):

        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out

class Backbone(BackboneBase):
    """ 
    Resnet backbone with frozen batchnorm
    """
    def __init__(self, name: str, train_backbone: bool, return_interim_layers: bool,
                dilation: bool):
        
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d
        ) # make pretrained true if requried 
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels=num_channels, return_interim_layers=return_interim_layers)


class Joiner(nn.Sequential):
    
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype)) # postional encoding
        
        return out, pos



