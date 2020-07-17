import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.utils import model_zoo

from collections import OrderedDict
import math

"""
Implementation of Resnet. Code taken and modified from PyTorch's Original 
implementation at https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 
"""

pretrained_models_url = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 Convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride 

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers=(3, 4, 6, 3), num_classes=1000, use_modify=True):

        super(ResNet, self).__init__()

        if not use_modify:
            
            print("WARNING: Not using modifications suggested to Resnet in the Paper. Pre-trained models may yeild")
            
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)

        else: 

            self.inplanes = 128

            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64, stride=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128, stride=1)
            self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # make layers here
        self.layer1 = self._make_resnet_layers(block, 64, layers[0])
        self.layer2 = self._make_resnet_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layers(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_resnet_layers(block, 512, layers[3], stride=1, dilation=4)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.use_modify = use_modify

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_resnet_layers(self, block, planes, num_block, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(
                
                nn.Conv2d(self.inplanes, planes*block.expansion,
                kernel_size=1, stride=stride, bias=False),
                
                nn.BatchNorm2d(planes*block.expansion),
                
                )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_modify:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.view(x.size(0), -1)
        x = self.fc(x)

        return x

# load pretrained model state dict
def  load_pretrained_weights(target, source):
    new_state_dict = OrderedDict()

    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source.items()):
        #print("target key: {} Source key: {}".format(k1, k2))
        if k2 in target.state_dict().keys():
            new_state_dict[k2] = v2
    target.load_state_dict(new_state_dict)


# some helper functions to easily build resnets
# Note: Pretraine = False because by default the Resnet
# class uses the modifications suggested by the PSPNet paper to support
# the pretrained models provided by the authors. Standard PyTorch implementation
# based pretrained model doenst support such a modification
# To avoid such modification, pass use_modify=False. Please note that way
# it wont support models provided by the author

def resnet18(pretrained=False, use_modify=True):
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], use_modify=use_modify)
    if pretrained:
        load_pretrained_weights(model, model_zoo.load_url(pretrained_models_url["resnet18"]))
    return model

def resnet34(pretrained=False, use_modify=True):

    model = ResNet(BasicBlock, [3, 4, 6, 3], use_modify=use_modify)
    if pretrained:
        load_pretrained_weights(model, model_zoo.load_url(pretrained_models_url["resnet34"]))
    return model

def resnet50(pretrained=False, use_modify=True):

    model = ResNet(Bottleneck, layers=[3, 4, 6, 3], use_modify=use_modify)
    if pretrained:
        load_pretrained_weights(model, model_zoo.load_url(pretrained_models_url["resnet50"]))
    return model

def resnet101(pretrained=False, use_modify=True):

    model = ResNet(Bottleneck, [3, 4, 23, 3], use_modify=use_modify)
    if pretrained:
        load_pretrained_weights(model, model_zoo.load_url(pretrained_models_url["resnet101"]))
    return model

def resnet152(pretrained=False, use_modify=True):

    model = ResNet(Bottleneck, [3, 8, 36, 3], use_modify=use_modify)
    if pretrained:
        load_pretrained_weights(model, model_zoo.load_url(pretrained_models_url['resnet152']))
    return model



        


