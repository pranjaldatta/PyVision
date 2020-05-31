import torch 
import torch.nn as nn  
import torch.nn.functional as F  

import os  
import numpy as np  

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, size, stride, padding=0):

        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, size, 
                    stride, padding, bias=False)
        # batch normalize values are defined the Sandberg Implementation
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu_fn = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu_fn(x)

        return x

class Block35_A(nn.Module):
    """
    Builds the 32x32 block. (Referred in the paper as Inception-
    Resnet-A)
    """
    def __init__(self, scale=1.0):

        super().__init__()

        self.scale = scale

        # now we construct the different branches. 
        # Refer to Inception-Resnet-A diagram in the paper
        self.branch0 = BasicConv2d(256, 32, 1, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1)
        )

        self.conv2d = nn.Conv2d(96, 256, 1, 1)
        self.relu_fn = nn.ReLU(inplace=False)
    
    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        x_cat = torch.cat((x0, x1, x2), 1)

        out = self.conv2d(x_cat)
        out = out * self.scale + x
        out = self.relu_fn(out)

        return out

class Block17_B(nn.Module):
    """
    Builds the 17x17 Block. (referred to as Inception-Resnet-B) 
    """
    def __init__(self, scale=1.0):
        
        super().__init__()
        
        self.scale = scale  
        
        self.branch0 = BasicConv2d(896, 128, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, 1, 1),
            BasicConv2d(128, 128, size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(256, 896, 1, 1)
        self.relu_fn = nn.ReLU(inplace=False)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        
        x_cat = torch.cat((x0, x1), 1)

        out = self.conv2d(x_cat)
        out = out * self.scale + x
        out = self.relu_fn(out)

        return out

class Block8_C(nn.Module):
    """
    Implements the 8x8 Block. (Referred to as Inception-Resnet-C in the paper) 
    """
    def __init__(self, scale=1.0, relu=True):

        super().__init__()
        
        self.scale = scale
        self.relu = relu

        self.branch0 = BasicConv2d(1792, 192, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, 1, 1),
            BasicConv2d(192, 192, (1, 3), 1, (0, 1)),
            BasicConv2d(192, 192, (3, 1), 1, (1, 0))
        )

        self.conv2d = nn.Conv2d(384, 1792, 1, 1)
        if self.relu:
            self.relu_fn = nn.ReLU(inplace=False)
    
    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x_cat = torch.cat((x0, x1), 1)
        out = self.conv2d(x_cat)
        out = out * self.scale + x
        if self.relu:
            out = self.relu_fn(out)
        
        return out


class Reduction_A(nn.Module):
    """
    Builds the Reduction A module. Refer to paper for details 
    """
    def __init__(self):

        super().__init__()

        self.branch0 = BasicConv2d(256, 384, 3, 2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, 1, 1),
            BasicConv2d(192, 192, 3, 1, 1),
            BasicConv2d(192, 256, 3, 2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        return out

class Reduction_B(nn.Module):
    """
    Builds Reduction B module. For more details check the paper 
    """
    def __init__(self):

        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, 1, 1),
            BasicConv2d(256, 384, 3, 2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, 1, 1),
            BasicConv2d(256, 256, 3, 2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, 1, 1),
            BasicConv2d(256, 256, 3, 1, 1),
            BasicConv2d(256, 256, 3, 2)
        )

        self.branch3 = nn.MaxPool2d(3, 2)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out