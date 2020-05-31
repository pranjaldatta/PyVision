import torch
import torch.nn as nn
import torch.nn.functional as F  

import gdown
import os
import numpy as np  
import json

from  ..utils.layer_factory import *


__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = ["vggface2", "casia-webface"]

class InceptionResnetV1(nn.Module):
    """
    Inception-Resnet-v1 serves as the embedding generator for this 
    Facenet implementation. It comes with weights pretrained on
    vggface2 and casia webface.

    """
    def __init__(self, pretrained='vggface2', wtspath="weights/", classify=True, num_classes=None, 
        dropout_probs=0.6, device="cpu"):

        super().__init__()

        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes
        self.wtspath = wtspath
        self.device = "cuda" if device is not "cpu" else "cpu"

        if self.pretrained is not None and self.pretrained not in __models__:
            raise NotImplementedError("only {} pretrained models supported but got {}".format(__models__, self.pretrained))
        if device is not "cpu" and not torch.cuda.is_available():
            raise ValueError("cuda not found but got device {}".format(device))
        elif self.pretrained == "vggface2":
            tmp_classes = 8631
        elif self.pretrained == "casia-webface":
            tmp_classes = 10575
        elif self.pretrained is None and self.num_classes is None:
            raise Exception("Both 'pretrained' and 'num_classes' cannot be None")
        elif self.pretrained is not None and wtspath is None:
            raise ValueError("wtspath cannot be none when pretrianed is ", self.pretrained)
        elif self.wtspath is None:
            raise ValueError("wtspath cannot be None")
        else:
            tmp_classes = self.num_classes
        
        # now we define the layers
        _repeat_num = [5, 10, 5] # number of times IR-A/B/C blocks repeat in each unit

        self.repeat_layers_0 = [ Block35_A(0.17) ] * _repeat_num[0] # Block A repeats 5 times
        self.repeat_layers_1 = [ Block17_B(0.10) ] * _repeat_num[1] # Block B repeats 10 times
        self.repeat_layers_2 = [ Block8_C(0.20) ] * _repeat_num[2]  # Block C repeats 5 times
        
        # now we define the base stem
        self.conv2d_1a = BasicConv2d(3, 32, 3, 2)
        self.conv2d_2a = BasicConv2d(32, 32, 3, 1)
        self.conv2d_2b = BasicConv2d(32, 64, 3, 1, 1)
        self.maxpool_3a = nn.MaxPool2d(3, 2)
        self.conv2d_3b = BasicConv2d(64, 80, 1, 1)
        self.conv2d_4a = BasicConv2d(80, 192, 3, 1)
        self.conv2d_4b = BasicConv2d(192, 256, 3, 2)

        # Inception-Resnet-A repeats x5
        # self.mixed_5a = nn.Sequential(*self.repeat_layers_0)
        self.repeat_1 = nn.Sequential(*self.repeat_layers_0)
        
        # Reduction Block A 
        # self.reduction_6a = Reduction_A()
        self.mixed_6a = Reduction_A()

        # Inception-Resnet-B repeats x10
        # self.mixed_6b = nn.Sequential(*self.repeat_layers_1)
        self.repeat_2 = nn.Sequential(*self.repeat_layers_1)
        
        # Reduction Block B
        # self.reduction_7a =Reduction_B()
        self.mixed_7a = Reduction_B()

        # Inception_Resnet-C repeats x5
        # self.mixed_8a = nn.Sequential(*self.repeat_layers_2)
        self.repeat_3 = nn.Sequential(*self.repeat_layers_2)

        self.block8 = Block8_C(relu=False)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_probs)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        
        if self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)
        else:
            self.logits = nn.Linear(512, tmp_classes)

        if pretrained is not None:
            resp = self.download(self.pretrained, self.wtspath)
            
            if resp == 1:
                print("Weights exist")
            else:
                print("Weights downloaded")
            
            self.load_weights(self.pretrained, self.wtspath)
        
        self.to(self.device)

    def download(self, pretrained, wtspath):
        """
        Downloads pretrained models. The pretrained models are hosted on 
        SRM-MIC's Google drive and are downloaded using gdown. Currently
        two pretrained models are supported: 
        -> VGGFace2
        -> Casia-Webface 
        """
            
        if os.path.exists(wtspath+"facenet-"+pretrained+".pt"):
            return 1
        elif os.path.exists(wtspath) and len(os.listdir(wtspath)) == 0:
            os.rmdir(wtspath)
        elif not os.path.exists(wtspath):
            os.mkdir(wtspath)
        
        config_path = os.path.dirname(__PREFIX__) + "/config/"
        with open(config_path + "weights_download.json") as fp:
            file_ids = json.load(fp)
         
        file_id = file_ids["facenet-"+pretrained+".pt"]
        url = 'https://drive.google.com/uc?id={}'.format(file_id)
        gdown.download(url, wtspath+"facenet-"+pretrained+".pt", quiet=False)

        return 0       
    
    def load_weights(self, pretrained, wtspath):
        """
        Load state dict.

        Arguments:
        - pretrained: pretrained model to use
        - wtspath: path where to look for weights 
        """
        state_dict_path = wtspath + "/facenet-" + pretrained + ".pt"
        try:
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)
        except Exception as exp:
            print("ERROR at model init: ", exp)
    
    def forward(self, x):

        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x








        



