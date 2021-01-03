import os 
import numpy as np 
import shutil
import cv2
from PIL import Image 
import sys 
import time 
import yaml
import gdown

import torch
import torch.nn as nn  
from torchvision import transforms

from .lib.model import EfficientDet
from .lib.utils import colors

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

sys.path.append(__PREFIX__)

import yaml
import json
import re



class EffdetInferAPI(object):

    def __init__(self, dataset='coco', thresh=0.4, gpu=False, common_size=512, verbose=False, wtspath="weights/", model_path=None):

        self.model_path = model_path
        self.verbose = verbose
        self.common_size = common_size
        self.thresh = thresh 

        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

        with open(__PREFIX__ + f"/config/dataset_{dataset}.yaml", "r") as f:
            config_file = yaml.safe_load(f)     

        self.class_list = config_file["class_list"]
        self.model_name = config_file['model_name']


        if gpu and not torch.cuda.is_available():
            raise ValueError(f"gpu not available but found gpu={gpu}")   
        self.device = "cuda" if gpu else "cpu"
        self.gpu = gpu

        
        #Instantiate the model
        self.model = EfficientDet(
            model_coeff = 0, 
            num_classes = len(self.class_list), 
            device = self.device
        )
        
        wtspath = wtspath+"{}.pth".format(self.model_name)
        resp = self._check_or_download_weights(__PREFIX__+"/"+wtspath)
        if resp == 0:
            print("weights downloaded.")
        else:
            print("weights found.")   
        
        if self.model_path is None:
            self.model_path = __PREFIX__+"/"+wtspath
        self.model.load_state_dict(torch.load(self.model_path))

        self.model = self.model.to(self.device)
    

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

    def detect(self, img):

        if isinstance(img, str):
            if os.path.exists(img):
                img_name = os.path.basename(img)
                img = cv2.imread(img)
            else:
                raise FileNotFoundError("2",img)
        elif isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image.Image):
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        orig_img = img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32) / 255.0 
        img = (img.astype(np.float32) - self.mean) / self.std
        height, width, _ = img.shape      

        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        img = cv2.resize(img, (resized_width, resized_height))  

        new_img = np.zeros((self.common_size, self.common_size, 3))
        new_img[0:resized_height, 0:resized_width] = img

        img = torch.from_numpy(img)

        start_time = time.time() 
        with torch.no_grad():
            img = img.to(self.device)
            scores, labels, boxes = self.model(img.permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale 
        duration = time.time() - start_time

        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy() 
        boxes = boxes.cpu().numpy()

        #try:
        
        to_delete = []
        if boxes.shape[0] > 0:

            for boxid in range(boxes.shape[0]):
                pred_probs = float(scores[boxid])
                #print(pred_probs)
                if pred_probs < self.thresh:
                    #print(f"small prob: {pred_probs}")
                    to_delete.append(boxid)
                    continue
                pred_labels = int(labels[boxid])
                xmin, ymin, xmax, ymax = boxes[boxid, :]
                
                color = colors(pred_labels)
                cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), color, 1)
                #print("drawing")
                put_text = self.class_list[pred_labels]+":%.2f"%pred_probs
                text_size = cv2.getTextSize(self.class_list[pred_labels]+":%.2f"%pred_probs, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        
                # clipping text boxes to prevent out-of-frame boxes
                text_x_max = xmin + text_size[0] + 3 if (xmin + text_size[0] + 3) < resized_width else resized_width
                text_y_max = ymin + text_size[1] + 4 if (ymin + text_size[1] + 4) < resized_height else resized_height
                
                xmin = int(xmin)
                ymin = int(ymin)
                text_x_max = int(text_x_max)
                text_y_max = int(text_y_max)

                cv2.rectangle(orig_img, (xmin, ymin), (text_x_max, text_y_max), color, -1)
                cv2.putText(
                    orig_img, put_text, (xmin, ymin + text_size[1] + 4), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
                    )

        
        scores = np.delete(scores, to_delete)
        labels = np.delete(labels, to_delete)
        boxes = np.delete(boxes, to_delete)
        
        
        labels = [self.class_list[label] for label in labels]

        return orig_img, duration, scores, labels, boxes

