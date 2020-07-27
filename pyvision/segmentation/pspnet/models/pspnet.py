import torch
import torch.nn as nn 
import torch.nn.functional as F  

from .backbone import * 

__extractors__ = {
    "resnet18" : resnet18,
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet101" : resnet101,
    "resnet152" : resnet152
}

class PPM(nn.Module):
    
    """The Pyramid Pooling Module""" 
    
    def __init__(self, input_dims, reduction_dims, scales):

        super(PPM, self).__init__()

        self.features = []
        for scale in scales:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(input_dims, reduction_dims, 1, bias=False),
                nn.BatchNorm2d(reduction_dims),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        result = [x]
        for feature in self.features:
            result.append(
                F.interpolate(feature(x), size=x_size[2:], mode="bilinear", align_corners=True)
            )

        result = torch.cat(result, 1)
        
        return result

class PSPNet_model(nn.Module):

    """ The main PSPNet Module""" 

    def __init__(self, extractor="resnet50", scales=[1,2,3,6], 
        dropout=0.1, num_classes=21, zoom_factor=8,
        criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):

        super(PSPNet_model, self).__init__()

        if len(scales)%4 != 0:
            raise ValueError("len of scales should be 4 but got ", len(scales))
        if num_classes <= 1:
            raise ValueError("num_classes should be > 1 but found ", num_classes)
        if zoom_factor not in [1, 2, 4, 8]:
            raise ValueError("zoom_factor should be in [1, 2, 4, 8] but got ", zoom_factor)

        self.extractor = extractor
        self.scales = scales
        self.dropout = dropout
        self.num_classes = num_classes
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.pretrained = pretrained

        backbone = __extractors__[self.extractor](False)
        
        # build the layers
        self.layer0 = nn.Sequential(
            backbone.conv1,           
            backbone.bn1,
            backbone.relu,
            backbone.conv2,
            backbone.bn2,
            backbone.relu,
            backbone.conv3,
            backbone.bn3, 
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        
        feature_dims = 2048 

        self.ppm = PPM(feature_dims, int(feature_dims/len(scales)), scales)
        feature_dims *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(feature_dims, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if not self.pretrained:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, self.num_classes, kernel_size=1)
            )
    

    def forward(self, x, y=None):
    
        x_size = x.shape

        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x) # for aux loss during training
        x = self.layer4(x_aux)

        x = self.ppm(x)
        
        x = self.cls(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        if not self.pretrained:
            aux = self.aux(x_aux)
            if self.zoom_factor != 1:
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)

            return x.max(1)[1], main_loss, aux_loss
        
        else:
            
            return x







