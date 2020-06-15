import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as T

from .detr import DETR_model, DETR_postprocess
from .models.backbone import Backbone, Joiner
from .models.transformers import Transformer
from .utils.position_encoding import PositionEmbeddingSine
from .utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .utils.box_utils import draw_box, clamp


import numpy as np  
from PIL import Image
import cv2  
import os 
import json
import pickle as pkl
import gdown
import time
import matplotlib.pyplot as plt

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = ["detr-resnet50", 
              "detr-resnet101",]


def available_models():
    return __models__


class DETR(object):

    def __init__(self, model="detr-resnet50", wtspath="weights/", conf_thresh=0.7,
                device="cpu", save=None, show=True, classfile="data/classes.txt", 
                colors="utils/pallete", annotate=True, pretrained=True):
        """
        The Detection Transformer (DETR) Module. One of the most interesting papers to come out 
        in 2020, this architecture uses a combination of a backbone CNN and a transformer to perform
        object detection in an end-to-end manner, unlike the conventional architectures wherein the 
        detection pipeline consists of a multitude of post-processing steps. 

        This is the model initialization module while the detection module is exposed as 'detect()'.

        Arguments:

            - model : Name of the model to be used. Currently two variants are supported - detr-resnet50
                      and detrs-resnet101
            
            - wtspath (default: weights/): path to pre-downloaded weights file.

            - conf_thresh (default: 0.7): confidence threshold for object detection

            - device (default: cpu): device to run inference on

            - save (default: None): path to save images with detections drawn in

            - show (default: True): show the images with detections drawn

            - classfile (default: data/classes.txt): path to file containing class names (must include N/As)

            - colors (default: utils/pallete): path to color pallete

            - pretrained (default: True): Use pretrained models.Currently training is not supported.

            - annotate (default: True): annotate classnames on detections if show/save if given

        """
        if model not in __models__:
            raise ValueError("{} not implemented".format(model))
        if device is not "cpu":
            if not torch.cuda.is_available():
                raise ValueError("cuda not available but got device=",device)
        if pretrained is not True:
            raise ValueError("Training not supported but got pretrained=", pretrained)
        if device is not "cpu" and device is not "gpu":
            raise ValueError("Unknown device value. Use device=gpu to run inference on gpu")
        if conf_thresh < 0.0 or conf_thresh > 1.0:
            raise ValueError("conf_thresh should be in range (0.0, 1.0) but found ",conf_thresh)
        
        self.model_name = model
        self.device = device
        self.save = save
        self.show = show
        self.device = "cuda" if self.device is not "cpu" else "cpu"
        self.conf_thresh = conf_thresh
        self.annotate = annotate

        self.class_names = self._load_classes(os.path.join(__PREFIX__, classfile))
        self.colors = self._load_pickle(os.path.join(__PREFIX__, colors))
    
        wtspath = wtspath+"{}.pth".format(self.model_name)
        resp = self._check_or_download_weights(__PREFIX__+"/"+wtspath)
        if resp == 0:
            print("weights downloaded.")
        else:
            print("weights found.")      
        
        try:
            self.model, self.postprocess = self._build_detr_model(conf_thresh=self.conf_thresh)
            self.model = self.model.eval()
        except:
            raise RuntimeError("error at model init")

        # now we load the pretrained model
        checkpoint = torch.load(self.wtspath)
        self.model.load_state_dict(checkpoint["model"])
        print("load complete")

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = self.model.to(self.device)

    def _load_classes(self, path):
        with open(path) as fp:
            class_names = fp.read().split("\n")[:-1]
        return class_names
    
    def _load_pickle(self, path):
        with open(path, "rb") as fp:
            colors = pkl.load(fp)
        return colors[:91]

    def _check_or_download_weights(self, wtspath):

        if os.path.join(__PREFIX__, "weights") not in wtspath and not os.path.exists(wtspath):
            raise FileNotFoundError("File not found. Either file doesnt exist or directory provided")
        elif not os.path.exists(wtspath):

            if os.path.exists(__PREFIX__+"/weights/") and len(os.listdir(__PREFIX__+"/weights/")) == 0:
                os.rmdir(__PREFIX__+"/"+"weights/")
                os.mkdir(__PREFIX__+"/weights/")
            
            if not os.path.exists(__PREFIX__+"/weights/"):
                os.mkdir(__PREFIX__+"/weights/")

            with open(os.path.join(__PREFIX__, "config/weights_download.json")) as fp:
                json_file = json.load(fp)
                print("fetching file ids for {}".format(self.model_name))
                file_id = json_file[self.model_name]
            
            url = 'https://drive.google.com/uc?id={}'.format(file_id)
            wtspath = __PREFIX__ + "/weights/{}.pth".format(self.model_name)
            gdown.download(url, wtspath, quiet=False)

            self.wtspath = wtspath

            return 0
        else:
            self.wtspath = wtspath
            return 1

    def _build_detr_model(self, conf_thresh=0.7, num_classes=91):
        
        print("building detr model..")
        model_details = self.model_name.split("-")
        backbone_name = model_details[1]
        try:
            if model_details[2] == "dc5":
                dilation = True
        except:
            dilation = False
        mask = False  # mask is true in panoptic models
        
        hidden_dims = 256
        backbone = Backbone(backbone_name, train_backbone=True, return_interim_layers=mask, dilation=dilation)
        position_encoding = PositionEmbeddingSine(hidden_dims // 2, norm=True)
        backbone_with_pos_enc = Joiner(backbone, position_encoding)
        transformer = Transformer(d_model=hidden_dims, return_intermediate_dec=True)

        backbone_with_pos_enc.num_channels = backbone.num_channels

        detr = DETR_model(backbone_with_pos_enc, transformer, num_classes, num_queries=100)
        postprocess = DETR_postprocess(conf=conf_thresh)

        return detr, postprocess

    
    def _load_state_dict(self, checkpoints):
        
        corrected_dict = {}
        for key, value in checkpoints.items():
            if "backbone" in key:
                key = key[:8] + key[10:]
                corrected_dict.update({key:value})
            else:
                corrected_dict.update({key:value})
        
        self.model.load_state_dict(corrected_dict)

    
    def _convert_to_tensor(self, x, batch_dim=True):
        
        orig_img = x
        if not isinstance(x, np.ndarray):
            #x = np.asarray(x)
            pass
       
        x = self.transform(x)

        x = torch.unsqueeze(x, 0)
        return orig_img, x
    

    def detect(self, img, show=None, save=None):
        """ 
        Run object detection on a given image. 

        Arguments:
            
            - img: a path to an img or an ndarray or PIL image

            - save (default: None): Overrides the "save" param at module init

            - show (default: None): Overrides the "show" param at module init

        Returns:

            - inference time and list of dicts containing score, labels, bounding box coords 
              in the format [..{"scores": ..., "labels": ..., "coords": ...}...]        
        """

        if isinstance(img, str):
            if os.path.exists(img):
                img_name = os.path.basename(img)
                #img = cv2.imread(img)
                img = Image.open(img)
                orig_img, img = self._convert_to_tensor(img)
            else:
                raise FileNotFoundError("2",img)
        elif isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image.Image):
            orig_img, img = self._convert_to_tensor(img)
        
        if save is None:
            save = self.save
        if show is None:
            show = self.show

        img_w, img_h = orig_img.size
        target_size = torch.Tensor([img_h, img_w])
        target_size = torch.unsqueeze(target_size, 0)
        
        img = torch.squeeze(img, 0) 
        img = nested_tensor_from_tensor_list([img])
        
        img = img.to(self.device)
        start_time = time.time()
        preds = self.model(img)
        end_time = time.time() - start_time

        results = self.postprocess(preds, target_size)
        
        #results = clamp(results, img_h, img_w) # giving wierd results

        if show:
            orig_img = np.array(orig_img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
            for item in results:

                label = self.class_names[item["labels"]]
                orig_img = draw_box(orig_img, item["coords"], label, item["labels"], self.colors, self.annotate)
        
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_img = Image.fromarray(orig_img)
            orig_img.show()

            if save is not None:
                orig_img.save(save)
            
            return end_time, results
        else:
            return end_time, results





