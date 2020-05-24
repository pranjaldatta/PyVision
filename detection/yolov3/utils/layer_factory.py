"""
we define the various blocks that are
used repetedly in the architecture
"""

import torch
import torch.nn as nn 
import numpy as np  
import torch.nn.functional as F  


class EmptyLayer(nn.Module):
    
    def __init__(self):
        """
        An empty layer class
        """
        super(EmptyLayer, self).__init__()


class MaxPool_stride1(nn.Module):

    def __init__(self, kernel_size, device):
        """
        Performs maxpool 2d operation with stride 1
        """
        super(MaxPool_stride1, self).__init__()
        self.size = kernel_size
        self.device = device
        self.pad = kernel_size - 1

    def forward(self, x):
        """
        forward pass function
        """
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.size, self.pad)(padded_x)

        return pooled_x


class DetectionLayer(nn.Module):
    def __init__(self, anchors, device):
        """
        The implementation of the custom yolo layer as mentioned 
        in the paper
        """
        super(DetectionLayer, self).__init__()
        
        self.device = device # add a device sanity check 
        self.anchors = anchors

    def forward(self, x, inp_dims, classes, conf):

        x = x.data  
        preds = predict_transform(x, inp_dims, self.anchors, classes, conf, self.device)
        return preds

class RegionLayer(nn.Module):
    
    def __init__(self, stride=2):
        super(RegionLayer, self).__init__()
        self.stride = stride
    
    def forward(self, x):
        
        assert(x.data.dim() == 4)
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0) , "Stride "+str(self.stride)+" not a proper divisor of height "+ str(H)
        assert(W % ws == 0) , "Stride "+str(self.stride)+" not a proper divisor of width "+ str(W)
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws)
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1,2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)

        return x


def make_pipeline(blocks, device):
    """
    build the yolo architecture from the parsed  cfg file

    Parameters:
    
    - blocks: a list of dictionaries containing information parsed
              from the config file
    
    Returns: 
    
    - a nn.ModuleList() object containing the modules

    """
    network_info = blocks[0]

    module_list = nn.ModuleList()

    index = 0 # since we need to implement skip connections

    in_filters = 3
    out_filters = []

    for x in blocks:

        module = nn.Sequential()

        if x["type"] == "net":
            continue
            
        if x["type"] == "convolutional":
            try:
                batch_norm = int(x["batch_normalize"])
                bias = False
            except:
                batch_norm = 0
                bias = True
            activation = x["activation"]
            filters = int(x["filters"])
            padding = int(x["pad"])
            size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                padding = (size-1)//2
            else:
                padding = 0
            
            # add the conv layer to the module list
            conv_layer = nn.Conv2d(in_filters, filters, size, stride=stride, padding=padding, bias=bias, dilation=1)
            module.add_module("conv_{0}".format(index), conv_layer)

            # check if batch norm layer is required. If yes, then add
            if batch_norm:
                bn_layer = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn_layer)
            
            # check if LeakyReLU activation is used
            if activation == "leaky":
                activ_fn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activ_fn)

        elif x["type"] == "upsample":
            stride = int(x['stride'])
            upsample_layer = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{0}".format(index), upsample_layer)

        elif x["type"] == "route":

            x["layers"] = x["layers"].split(",")

            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index
            
            empty_layer = EmptyLayer()
            module.add_module("route_{0}".format(index), empty_layer)

            # note: start and end always has to be negative because
            # being negative denotes that the required layer has already 
            # been encountered. Postive start and/or end at this point
            # means that the layer hasnt been encountered relative to 
            # index.

            if end < 0:
                filters = out_filters[index + start] + out_filters[index + end]
            else:
                filters = out_filters[index + start]
        
        elif x["type"] == "shortcut":
            
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif x["type"] == "maxpool":
            
            stride = int(x["stride"])
            size = int(x["size"])

            if stride != 1:
                maxpool_layer = nn.MaxPool2d(size, stride)
            else:
                maxpool_layer = MaxPool_stride1(size, device)
        
            module.add_module("maxpool_{}".format(index), maxpool_layer)
        
        elif x["type"] == "yolo":
            
            masks = x["mask"].split(",")
            masks = [int(mask) for mask in masks]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in masks]

            yolo_layer = DetectionLayer(anchors, device)

            module.add_module("Detection_{0}".format(index), yolo_layer)

        else:
            raise NotImplementedError("layer {} not implemented".format(x["type"]))
        
        module_list.append(module)
        in_filters = filters
        out_filters.append(filters)
        index += 1

    return (network_info, module_list)