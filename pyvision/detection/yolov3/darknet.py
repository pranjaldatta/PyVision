import torch
import torch.nn as nn
import torch.nn.functional as F  

import numpy as np      
import cv2
import matplotlib.pyplot as plt  
import os
import gdown
import json

from .utils.layer_factory import make_pipeline
from .utils.parse_config import parse_config
from .utils.utils import predict_transforms

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = open(__PREFIX__ + "/config/models_supported.txt").read().split("\n")


class Darknet(nn.Module):
    """
    Here we define the darknet class.

    Parameters:
    - cfg_path: path to the config file
        
    - pretrained: Default is set True. If true than in absence of 
                    weights file in weights/ directory, it downloads
                    pretrained default weights. If False, 
                    throws an error

    - device: device to run the model on.
                default is "cpu"
    """
    def __init__(self, model_name, cfg_path, pretrained=True, device="cpu"):

        super(Darknet, self).__init__()

        cfg_path = __PREFIX__ + "/config/{}.cfg".format(model_name)
        self.WEIGHTS_PATH = __PREFIX__ + "/weights/{}.weights".format(model_name)
        
        self.block_list = parse_config(cfg_path)
        self.device = device
        self.net_info, self.model = make_pipeline(self.block_list, self.device)

        # stores info regarding pretrained models.
        # check load_weights function for docs
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        # check if weights file exist or not
        # if not download
        
        resp = self.download_weights(model_name, self.WEIGHTS_PATH, pretrained)
        if resp == 0:
            print("Weight download complete.")
        elif resp == 1:
            print("Weight file exists.")
    
    
    def get_blocklist(self):
        return self.block_list
    
   
    def get_model(self):
        return self.model
    
    
    def get_netinfo(self):
        return self.net_info

    
    def download_weights(self, model="yolov3", wtspath="weights/yolov3.weights", pretrained=True):
        """
        downloads the pretrained weights and places them in the
        yolov3/weights directory. 

        check documentation or available_models() for supported architetures.

        The pretrained weights are hosted at SRM-MIC's Google Drive and are download
        with gdown
        """

        if os.path.exists(wtspath):
            return 1
        
        if os.path.exists(__PREFIX__+"/weights/") and len(os.listdir(__PREFIX__+"/weights")) == 0:
            os.rmdir(__PREFIX__+"/weights/")
        
        if not os.path.exists(__PREFIX__+"/weights/"):
            os.mkdir(__PREFIX__+"/weights/")
        
        if pretrained:
            with open(__PREFIX__+"/config/weights_download.json") as fp:
                json_file = json.load(fp)
                print("fetching file id for {}.weights".format(model))
                file_id = json_file["{}.weights".format(model)]
                       
            url = 'https://drive.google.com/uc?id={}'.format(file_id)
            gdown.download(url, wtspath, quiet=False)
            return 0
        else:
            raise FileNotFoundError("Weight file not found.")
            

    def load_weights(self):
        """
        Loads the weights into the model.

        By Default, looks for the weights in the weights/ 
        directory.
        """
        fp = open(self.WEIGHTS_PATH, "rb")

        # the first 4 values are information regarding
        # 1.Major Version 2. Minor Version 3. Subversion No
        # 4.Images Seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.model)):

            mod_type = self.block_list[i+1]["type"]

            if mod_type == "convolutional":
                
                module = self.model[i]
                try:
                    batch_norm = int(self.block_list[i+1]["batch_normalize"])
                except:
                    batch_norm = 0
                
                conv = module[0]
                
                if batch_norm:
                    bn = module[1]

                    num_bn_bais = bn.bias.numel()

                    # load the weights for batch norm
                    bn_bias = torch.from_numpy(weights[ptr : ptr+num_bn_bais])
                    ptr += num_bn_bais

                    bn_weights = torch.from_numpy(weights[ptr : ptr+num_bn_bais])
                    ptr += num_bn_bais

                    bn_running_mean = torch.from_numpy(weights[ptr : ptr+num_bn_bais])
                    ptr += num_bn_bais

                    bn_running_var = torch.from_numpy(weights[ptr : ptr+num_bn_bais])
                    ptr += num_bn_bais
                    
                    # cast the weight shapes to be like the module shapes
                    bn_bias = bn_bias.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    
                    # copy the weights to the layer
                    bn.bias.data.copy_(bn_bias)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    
                    # now we load the weights of the biases of the conv layer
                    
                    # num of biases in this conv layer
                    num_bias = conv.bias.numel()
                    
                    # load the weights
                    conv_bias_weights = torch.from_numpy(weights[ptr:ptr+num_bias])
                    ptr += num_bias
                    
                    # cast them to the required shape
                    conv_bias_weights = conv_bias_weights.view_as(conv.bias.data)

                    # copy them into the layer biases
                    conv.bias.data.copy_(conv_bias_weights)

                # now we load the conv layer weights
                num_weights = conv.weight.numel()

                # load the weights
                conv_weights = torch.from_numpy(weights[ptr : ptr+num_weights])
                ptr += num_weights

                # cast them to the required shape
                conv_weights = conv_weights.view_as(conv.weight)

                # copy them
                conv.weight.data.copy_(conv_weights)

    def forward(self, x):
        
        detections = []
        modules = self.block_list[1:]
        outputs = {} # caching for route layers
        
        write = 0
        for i in range(len(modules)):
            
            mod_type = modules[i]["type"]
            
            if mod_type == "convolutional" or mod_type == "upsample" or mod_type == "maxpool":
                
                x = self.model[i](x)
                outputs[i] = x               

            elif mod_type == "shortcut":
                
                _from = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i + _from]
                outputs[i] = x

            elif mod_type == "route":
                
                layers = modules[i]["layers"]
                layers = [int(layer) for layer in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    
                    _feature_map1 = outputs[i + layers[0]]
                    _feature_map2 = outputs[i + layers[1]]

                    x = torch.cat((_feature_map1, _feature_map2), 1)
                
                outputs[i] = x
            
            elif mod_type == "yolo":
                
                anchors = self.model[i][0].anchors

                input_dims = int(self.net_info["height"])

                num_classes = int(modules[i]["classes"])

                x = x.data 
                
                x = predict_transforms(x, input_dims, anchors, num_classes, device=self.device)
                
                if type(x) == int:
                    continue

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                
                outputs[i] = outputs[i-1]

        try:
            return detections
        except:
            return 0        
