import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import cv2
from PIL import Image
import os 
import gdown 
import numpy as np  
import json

from .models.backbone import * 
from .models.pspnet import PSPNet_model
from .util.utils import make_color_seg_map

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = ["pspnet-resnet50-voc2012",
              "pspnet-resnet101-voc2012",
              "pspnet-resnet50-ade20k",
              "pspnet-resnet50-cityscapes"
             ]

def available_models():
    """ Returns list of all supported models """
    return __models__

class PSPNet(object):

    def __init__(self, model="pspnet-resnet50-voc2012", wtspath="weights/",
                device="cpu", save=None, show=True, draw_map=True, draw_blend=True,
                classfile=None, colors=None, scales=[1, 2, 3, 6], psp_size=2048, 
                deep_features_size=1024, blend_alpha=.7, downsize=None, pretrained=True):

        """
        Pyramid Scene Parsing (PSPNet) Segmentation Module. This class exposes the inference method that
        is used to run inference on an Image. For implementation details refer
        to the PSPNet readme.md . All major model parameters can be confifured here.

        Returns: 

            preds, color_img(optional), blend_img (optional)

        -  preds (numpy array of shape (225x225) or (473x473)): A numpy array whose every pixel contains 
                index of the class that pixel is classified into. Whether it is of dims 
                225x225 or 473x473 is determined by the "downsize" param. Check readme.md.
        
        - color_img (optional; PIL Image): a PIL image of the segmentation map

        - blend_img (optional; PIL Image): a PIL image of the segmentation map blended into the 
                                           original image.
        

        Arguments: 

        - model (default: pspnet-resnet50-voc2012): The pretrained model to be used. 
                                                    For list of all supported models,
                                                    either check readme.md or available_models()
        
        - wtspath (default: weights/): Path to .pth file. NOTE: Provide full path to .pth if providing
                                       custom path. Else, leave the parameter unchanged, for the module 
                                       will automatically download the required weights in default directory

        - device (default: cpu): device to run inference on.

        - save (default: None): full path + filename that the results are to be saved as. For example,
                                to save a result image as path/to/result.png, save should be path/to/result.
        
        - show (default: True): Show the results 

        - draw_map (default: True): Draw the segmentation map. 

        - draw_blend (default: True): Blend the segmentaion map into the original input image and 
                                      produce a new image
        
        - classfile (default: None): path to classfile. No need to provide any value as classfile is already
                                    available in repo
        
        - colors (default: None): path to color pallette file. No need to provide any value as color pallette files
                                  are already available in repo.

        - scales (default: [1, 2, 3, 6]): Scale factors to be used in the PPM Module. For more details refer to paper. 

        - blend_alpha (default: .7): alpha channel parameter of segmentation map that is blended into original input image

        - downsize (default: None): Read readme.md for more information. Basically, when running without gpu, running inference 
                                    with default input image size of 473x473 is computationally intensive so we downsize to 225x225.
                                    When downsize=None, the model automatically downsizes the image to 225x225 in the absence of a gpu but 
                                    in the presence of one, it automatically uses the recommeneded size of 473x473. downsize can be made 
                                    True or False to enforce either behavior.

        - pretrianed (default: True): To whether use pretrained models or not. Currently the only method supported.
        """


        if model not in __models__:
            raise ValueError("{} not supported yet. Check PSPNet.available_models() for supported models".format(model))
        if device is not "cpu":
            if not torch.cuda.is_available():
                raise ValueError("device={} but cuda not detected".format(device))
            device = "cuda"
        if classfile is not None and not os.path.exists(classfile):
            raise FileNotFoundError("{} not a valid path".format(classfile))
        if colors is not None and not os.path.exists(colors):
            raise FileNotFoundError("{} not a valid path".format(colors))

        self.device = device
        self.model_name = model
        self.save = save 
        self.show = show  
        self.draw_map = draw_map 
        self.wtspath = wtspath
        self.pretrained = pretrained
        self.blend_alpha = blend_alpha
        self.draw_blend = draw_blend

        value_scale = 255
        self.std = None
        self.mean = [item * value_scale for item in [0.485, 0.456, 0.406]]
        self.std = [item * value_scale for item in [0.229, 0.224, 0.225]]
        self.downsize = None

        if downsize is None and self.device is "cpu":
            self.downsize = True
        elif downsize is False and self.device is "cpu":
            print("Running with default input image size of 443 on cpu is computationally intensive. Change it to False or default None to use a smaller 225x225 image for input")
        
        #note add checks for the below
        self.scales = scales
        self.psp_size = psp_size
        self.deep_feat_size = deep_features_size

        _tmp = self.model_name.split("-")
        _backbone, _dataset_name = _tmp[1], _tmp[2]

        with open(__PREFIX__+"/data/{}_classes.txt".format(_dataset_name), "r") as f:
            self.class_names = f.read().split("\n")[:-1]
        with open(__PREFIX__+"/data/{}_colors.txt".format(_dataset_name), "r") as f:
            self.colors = f.read().split("\n")[:-1]   
        self.colors = np.loadtxt(__PREFIX__+"/data/{}_colors.txt".format(_dataset_name)).astype(np.uint8)     
        
        # now we initialize the model
        self.model = PSPNet_model(extractor=_backbone, scales=self.scales, num_classes=len(self.class_names)+1)
        
        resp = self._check_or_download_weights(__PREFIX__+"/"+wtspath)
        #resp = self._check_or_download_weights(wtspath)
        print(self.wtspath)
        if resp == 0:
            print("Weights downloaded.")
        else:
            print("Weights found.")
        
        # loading state dict
        if self.pretrained:
            self.model = self.model.to(self.device)          
            self._load_weights()
            self.model.eval()
            

    def _check_or_download_weights(self, wtspath):

        if os.path.join(__PREFIX__, "weights") not in wtspath and not os.path.exists(wtspath):
            raise FileNotFoundError("File not found. Either file doesnt exist or directory provided")
        elif not os.path.exists(wtspath + self.model_name + ".pth"):
            
            if os.path.exists(__PREFIX__+"/weights/") and len(os.listdir(__PREFIX__+"/weights/")) == 0:
                os.rmdir(__PREFIX__+"/"+"weights/")
                os.mkdir(__PREFIX__+"/weights/")

            if not os.path.exists(__PREFIX__+"/weights/"):
                os.mkdir(__PREFIX__+"/weights/")

            with open(os.path.join(__PREFIX__, "config/weights_download.json")) as fp:
                json_file = json.load(fp)
                print("Fetching file ids for {}".format(self.model_name))
                file_id = json_file[self.model_name]

            url = 'https://drive.google.com/uc?id={}'.format(file_id)
            wtspath = __PREFIX__ + "/weights/{}.pth".format(self.model_name)
            gdown.download(url, wtspath, quiet=False)

            self.wtspath = wtspath

            return 0 
        
        else:
            self.wtspath = wtspath + "{}.pth".format(self.model_name)
            return 1


    def _load_weights(self):
        
        print("loading")

        new_state_dict = {}
        not_filled = [] #for debugging
        
        source_state_dict = torch.load(self.wtspath, map_location=torch.device(self.device))
        target_state_dict = self.model.state_dict()

        count = 0
        for (k, v) in source_state_dict["state_dict"].items():
            
            if k[7:] in target_state_dict.keys():
                    
                new_state_dict[k[7:]] = v
                count += 1
            else:
                not_filled.append(k[7:])

        if len(target_state_dict) != count:
            print("Some params of state dict werent filled. Not filled keys are")
            print(not_filled)
            raise Exception("all target state dict params not satisfied")
        
        else:
            self.model.load_state_dict(new_state_dict)

    
    def inference(self, img, save=None, show=None, draw_map=None, draw_blend=None):
        

        if save is None:
            save = self.save
        if show is None:
            show = self.show
        if draw_map is None:
            draw_map = self.draw_map
        if draw_blend is None:
            draw_blend = self.draw_blend
        
        if draw_blend is True and draw_map is False:
            raise ValueError("draw_blend cannot be True with draw_map being False")
        if show is True and draw_map is False:
            raise ValueError("show cannot be True with draw_map being False")
        

        if isinstance(img, str):
            if os.path.exists(img):
                
                img_name = os.path.basename(img)
                img = Image.open(img).convert("RGB")

                orig_w, orig_h = img.size
                if self.downsize is True:
                    
                    new_h = 225
                    new_w = 225

                else:

                    new_h = 473
                    new_w = 473    
                
                img = img.resize((new_w, new_h))

                orig_img = img
                img = np.array(img)
            else:
                raise FileNotFoundError("2",img)
        
        elif isinstance(img, np.ndarray):

            if self.downsize is True:
                new_h, new_w = 225, 225
            elif self.downsize is False:
                new_h, new_w = 473, 473

            img = cv2.resize(img, (new_h, new_w))
        
        elif isinstance(img, Image.Image):
            
            img = img.convert('RGB')
                
            if self.downsize is True:                    
                new_h = 225
                new_w = 225
            else:
                new_h = 473
                new_w = 473    
                
            img = img.resize((new_w, new_h))

            orig_img = img
            img = np.array(img)
            
     
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        
        if self.std is None:
            for t, m in zip(img, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)    

        img = img.unsqueeze(0).to(self.device)

        preds = self.model(img)
        
        preds = F.softmax(preds, dim=1)
        preds = preds[0]
        preds = preds.data.cpu().numpy()
        preds = preds.transpose(1, 2, 0)
        preds = np.argmax(preds, axis=2)

        
        if draw_map:            
            color_img = make_color_seg_map(preds, self.colors)
        
        if draw_blend:
            
            if isinstance(orig_img, np.ndarray):
                orig_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            
            orig_img = orig_img.convert("RGBA")
            color_img = color_img.convert("RGBA")
            blend = Image.blend(orig_img, color_img, alpha=.7)


        if save is not None:
            color_img.save("{}_map.png".format(save))
            
            try:
                blend.save("{}_blend.png".format(save))
            except:
                pass

        if show is not None:
            color_img.show()   

            try:
                blend.show()
            except:
                pass     

        try:
            return preds, color_img, blend
        except:
            try:
                return preds, color_img
            except:
                return preds




