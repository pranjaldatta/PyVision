import torch  
import torch.nn as nn 
import torch.nn.functional as F  
from torchvision import transforms

import os 
import numpy as np
import json
import gdown
from PIL import Image  
import cv2  

from .models.fcn_net import _build_fcn
from .util.utils import make_color_seg_map

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = ["fcn-resnet50-coco", "fcn-resnet101-coco"]

def available_models():
    """Returns list of available models""" 
    return __models__

class FCN(object):

    def __init__(self, model="fcn-resnet50-coco", wtspath="weights/", device="cpu",
            save=None, show=True, draw_map=True, draw_blend=True, classfile=None, 
            colors=None, blend_alpha=.7, pretrained=True):

        """ 
        Pyramid Scene Parsing (PSPNet) Segmentation Module. This class exposes the inference method that
        is used to run inference on an Image. For implementation details refer
        to the PSPNet readme.md . All major model parameters can be configured here.

        Returns: 
            preds, segmentation_map (optional), blend_img (optional)
        
        -  preds (numpy array of shape (H, W)): A numpy array whose every pixel contains 
                index of the class that pixel is classified into. (H, W) are height and 
                width of the given image respectively. Check readme.md.
        
        - color_img (optional; PIL Image): a PIL image of the segmentation map
        
        - blend_img (optional; PIL Image): a PIL image of the segmentation map blended into the 
                                           original image.

        Arguments: 
        
        - model (default: fcn-resnet50-coco):   The pretrained model to be used. 
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
        
        - blend_alpha (default: .7): alpha channel parameter of segmentation map that is blended into original input image
              
        - pretrained (default: True): To whether use pretrained models or not. Currently the only method supported.
        
        """ 
        if model not in __models__:
            raise ValueError(f"{model} not supported yet. Only {__models__} supported")
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
        self.draw_blend = draw_blend
        self.draw_map = draw_map
        self.classfile = classfile
        self.colors = colors
        self.blend_alpha = blend_alpha
        self.pretrained = pretrained

        self._norm_mean = [0.485, 0.456, 0.406]
        self._norm_std = [0.229, 0.224, 0.225]

        self.tsfms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=self._norm_mean, std=self._norm_std)
        ])

        _dataset_name = "voc2012"
        with open(__PREFIX__+"/data/{}_classes.txt".format(_dataset_name), "r") as f:
            self.class_names = f.read().split("\n")[:-1]
        with open(__PREFIX__+"/data/{}_colors.txt".format(_dataset_name), "r") as f:
            self.colors = f.read().split("\n")[:-1]   
        self.colors = np.loadtxt(__PREFIX__+"/data/{}_colors.txt".format(_dataset_name)).astype(np.uint8)     

        _backbone = self.model_name.split("-")[1]
        
        self.model = _build_fcn(_backbone, 21, pretrained=False, aux=True)

        resp = self._check_or_download_weights(__PREFIX__+"/"+wtspath)
        #resp = self._check_or_download_weights(wtspath)
        print(self.wtspath)
        if resp == 0:
            print("Weights downloaded.")
        else:
            print("Weights found.")

        if self.pretrained:
            self._load_weights(self.wtspath)
            print("Model load complete.")
            self.model.eval()
        
        self.model = self.model.to(self.device)

        

    def class_names(self):
        return self.class_names


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
    
    def _load_weights(self, wtspath):

       
        source_state_dict = torch.load(self.wtspath, map_location=torch.device(self.device))

        self.model.load_state_dict(source_state_dict)
     

    def inference(self, img, save=False, show=False, draw_blend=None, draw_map=None):

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
            else:
                raise FileNotFoundError(f"{img} no such path")
        elif isinstance(img, Image.Image):
            pass 
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        orig_img = img
        img = self.tsfms(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.model(img)["out"]
            output = torch.squeeze(output, 0).cpu().numpy()
            output = np.argmax(output, axis=0)
        
        if draw_map:
            color_img = make_color_seg_map(output, self.colors)
        
        if draw_blend:

            orig_img = orig_img.convert("RGBA")
            color_img = color_img.convert("RGBA")
            blend = Image.blend(orig_img, color_img, alpha=.7)

        if save is not None: 

            color_img.save(f"{save}_map.png")

            try:
                blend.save(f"{save}_blend.png")
            except:
                pass 
        
        if show is not None and show is True:

            try:
                blend.show()
            except:
                pass     

        try:
            return output, color_img, blend
        except:
            try:
                return output, color_img
            except:
                return output

