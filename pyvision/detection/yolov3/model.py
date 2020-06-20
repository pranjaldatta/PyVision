import torch
import torch.nn as nn
import torch.nn.functional as F  

import sys
import numpy as np  
import cv2
import os
from PIL import Image
import time
import pickle

from .utils.utils import postprocess, load_classes
from .darknet import Darknet
from .utils.preprocess import prepare_img_cv2
from .utils.box_utils import draw_box


__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

__models__ = open(__PREFIX__+"/config/models_supported.txt").read().split("\n")

def available_models():
    """available_models : lists the models currently supported

    Returns
    -------
    list:
        list of models supported
    """
    return __models__

class YOLOv3:

    """
    YOLOv3 detection class. This class exposes the detect() method used for
    running inference. All the major network parameters can be defined here.
    Defaults are set but can be customized.

    Use 'model.__models__' to check which architectures are defined   

    Parameters:

    - save (default: None): Location to save image with detections in. Doesnt save if None.
    
    - show (default: True): show the image with the detections

    - draw (default: True): draw the boxes and return image with boxes drawn

    - device (default: cpu): device to run inference on

    - model (default: yolov3): default model to run. (use available_models() to see supported models)

    - cfg (default: config/yolov3.cfg): config file for yolo architecture. default yolov3 architecture supported. More to be added later.

    - classfile (default: data/coco.names): class names

    - confidence (default: 0.5): minimum confidence for object detection

    - nms_thresh (default: 0.4): threshold for non max suppression

    - reso (default: 416): default resolution of input image

    - scales (default: [1,2,3]): default scales for detection
        
    """

    def __init__(self, save=None, show=True, device="cpu", model="yolov3", 
        cfg="config/yolov3.cfg", classfile="data/coco.names", colors="utils/pallete", 
        draw=True, confidence=0.5, nms_thresh=0.4, reso=416, scales=[1,2,3]):     
        

        if save is None and show is None:
            raise TypeError("save and show cannot be concurrently None")
    
        if save is not None and not os.path.exists(save):
            os.mkdir(save)
    
        if device == "gpu":
            if not torch.cuda.is_available():
                raise Exception("CUDA not available but received device = 'gpu'")
            device = "cuda"
        else:
            device = "cpu"

        self.save = save
        self.show = show
        self.device = device 
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.resp = reso
        self.scales = scales
        self.classes = load_classes(__PREFIX__+"/"+classfile)
        self.colors = pickle.load(open(__PREFIX__+"/"+colors, "rb"))
        self.draw = draw

        # now set the model to be used. The available models
        # are defined in __models__
        if model not in __models__:
            raise NotImplementedError("{} model has not been supported yet".format(model))


        # now we initialize the model 
        try:
            self.model = Darknet(model, cfg, device=self.device)
            self.model.load_weights()
        except Exception as exp:
            print("ERROR at model init: ", exp)
            exit()

        self.model.eval()
        self.model = self.model.to(device)
    

    def detect(self, img, save=None, show=None, draw=None):
        r"""
        The main method to run yolov3 inference on a given image.

        Arguments:

        -img: can be cv2 or PIL image or even a path

        -save (default: None): location to save the detection in.Overides values set during class initialization. if not None.
                    
        -show (default: None): whether to show the detection. Overides value set during class initialization.

        -draw (default: None): whether to show the image with boxes drawn. Overides value set during class initialization.

        Return:

        - a tuple of the image on which the bouding boxes are drawn and a list of
          dictionaries containing all detection info 
        
        - returned as (time_taken, img, list) or (time_taken, list)

        - img (dtype: np.ndarray): image on which detections are drawn.
        
        - list (dtype: list): list of dictionary containing all information
                               regarding detections made. Dicts are in the 
                               format {"class": ... , "coords": ... , "score": ...}


        """
    
        if save is not None and not os.path.exists(save):
            os.mkdir(save)
        
        if save is None:
            save = self.save
        if show is None:
            show = self.show   
        if draw is None:
            draw = self.draw
   

        if isinstance(img, str):
            if os.path.exists(img):
                img_name = os.path.basename(img)
                img = cv2.imread(img)
            else:
                raise FileNotFoundError("2",img)
        elif isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image.Image):
            img = np.array(img)
            img = img[:,:,::-1]

        # add more checks later
    
        net_info = self.model.get_netinfo()  
        net_dims = int(net_info["height"])

        img, orig_img, orig_dims = prepare_img_cv2(img, net_dims)
        orig_dims = torch.FloatTensor(orig_dims).repeat(1, 2)
        #orig_dims = orig_dims.to(self.device)
        img = img.to(self.device)
        objs = []

        start_time = time.time()

        with torch.no_grad():
            preds = self.model(img)
        
        #if self.device is "cuda:0" and torch.cuda.is_available():
            #preds = preds.cpu()
        preds = postprocess(preds, self.device, self.confidence, len(self.classes))

        if type(preds) == int:
            return None, []

        # the detected classes
        det_cls = [self.classes[int(detection[7])] for detection in preds]

        if self.device is not "cpu":
            torch.cuda.synchronize()
        
        if self.device is not "cpu":
            preds = preds.cpu()

        orig_dims = torch.index_select(orig_dims, 0, preds[:,0].long())

        scaling_factor = torch.min(net_dims/orig_dims, 1)[0].view(-1, 1)


        # basically we are rescaling the bounding box coords wrt to unpadded
        # original image
        preds[:, [1,3]] -= (net_dims - scaling_factor*orig_dims[:, 0].view(-1, 1))/2
        preds[:, [2,4]] -= (net_dims - scaling_factor*orig_dims[:, 1].view(-1, 1))/2
        preds[:, 1:5] /= scaling_factor

        # now, since the box coords are based on the padded images, it may so happen
        # that some box coords are outside the edges of the unpadded image. Hence we
        # clip such coords to the edge of the unpadded images
        for i in range(preds.shape[0]):
            preds[i, [1, 3]] = torch.clamp(preds[i, [1, 3]], 0.0, orig_dims[i, 0])
            preds[i, [2, 4]] = torch.clamp(preds[i, [2, 4]], 0.0, orig_dims[i, 1])

        end_time = time.time()

        if draw:
            result = list(map(lambda pred, _cls: draw_box(pred, orig_img, _cls, self.colors), preds, det_cls))[0]

        # we also generate a list of dictionary object that stores all necessary information
        # and we return it so it can be furthur used
        # info stored as [..{'class':  , 'coords': , 'score': }]
        for idx, pred in enumerate(preds):
            _cls = det_cls[idx]
            _coords = [c.int().item() for c in pred[1:5]]
            _score = pred[5].float().item()
            objs.append({"class":_cls, "coords":_coords, "score": _score})


        if show:
            cv2.imshow("Detection", result)
            if cv2.waitKey() == ord('q'):
                cv2.destroyAllWindows()
        
        if save is not None and save is not False:
            
            if draw is False:
                result = list(map(lambda pred, _cls: draw_box(pred, orig_img, _cls, self.colors), preds, det_cls))[0]
            try:
                img_name
                img_name = "det_"+img_name
            except:
                # no img name found
                img_name = "detection.jpg"
            cv2.imwrite(os.path.join(save, img_name), result)

        time_taken = end_time - start_time

        if draw:
            return time_taken, result, objs
        else:
            return time_taken, objs