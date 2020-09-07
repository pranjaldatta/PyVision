import torch  
import torch.nn as nn  
import torch.nn.functional as F   
from torchvision.models._utils import IntermediateLayerGetter

from .backbone import resnet101, resnet50

from collections import OrderedDict

__backbones__ = {
    "resnet50" : resnet50,
    "resnet101" : resnet101
}

def _build_fcn(name, num_classes, aux, pretrained=False):
    
    backbone = __backbones__[name](
        pretrained=pretrained, 
        replace_stride_with_dilation=[False, True, True]
    )

    final_layers = {'layer4': 'out'}
    if aux:
        final_layers['layer3'] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=final_layers)

    aux_classifier = None 
    if aux:
        inplanes = 1024 
        aux_classifier = FCNHead(inplanes, num_classes)
    
    inplanes = 2048 
    classifier = FCNHead(inplanes, num_classes)
    #base_model = FCNModel()

    fcn_model = FCNModel(backbone, classifier, aux_classifier)

    return fcn_model


class FCNModel(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):

        super(FCNModel, self).__init__()
        
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
    
    def forward(self, x):

        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result["aux"] = x
        
        return result


class FCNHead(nn.Sequential):

    def __init__(self, inchannels, channels):
        
        intermediate_channels = inchannels // 4 
        layers = [
            nn.Conv2d(inchannels, intermediate_channels, 3, padding=1, bias=False), 
            nn.BatchNorm2d(intermediate_channels), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Conv2d(intermediate_channels, channels, 1)
        ]
    
        super(FCNHead, self).__init__(*layers)


