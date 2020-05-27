import numpy as np
from PIL import Image
import os
import json
import gdown
import cv2

import torch

from .nets import PNet, RNet, ONet
from .stage_one import first_stage
from .stage_two import get_image_boxes
from .utils.visualize import show_boxes
from .utils.utils import nms, convert_to_square, calibrate_boxes

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

class MTCNN:
    """
    method that accepts an image and returns bounding boxes around faces

    Parameters:
    -> min_face_size (float): minimum size of face to look for
    -> conf_thresh (list): list of confidence thresholds for various parts
                           parts in the pipeine. (Size = 3)
    -> nms_thresh (list): list of overlap thresholds for nms (size = 3)
    """
    def __init__(self, device="cpu", min_face_size = 20.0, conf_thresh=[0.7, 0.7, 0.8], nms_thresh=[0.7, .7, .7], pretrained=True):

        if device is not "cpu":
            raise NotImplementedError("gpu support not implemented. cpu only.")
        if len(conf_thresh) != 3 or len(nms_thresh) != 3:
            raise AssertionError("conf_thresh or nms_thresh of len :{},{} while expected size: 3".format(len(conf_thresh), len(nms_thresh)))
        if min_face_size <= 0.0 or min_face_size is None:
            raise ValueError("min_face_size expected > 0.0 . Found {}".format(min_face_size))
        if not pretrained:
            raise NotImplementedError("Only Inference supported. Found pretrained=", pretrained)
        
        self.min_face_size = min_face_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pretrained = pretrained
        self.weights_path = __PREFIX__ + "/weights/"


        if pretrained:
            resp = self.download_weights(self.weights_path)
            if resp == 1:
                print("Weight files exist.")
            elif resp == 0:
                print("Weight files downloaded.")
        
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        
            
    def download_weights(self, wtspath):
        """download_weights 

        Parameters
        ----------
        wtspath : [str; path]
            [the full path to the ./weights/ folder]

        Returns
        -------
        [int]
            [1 : weights already exist
             2 :  weights have been downloaded successfully]
        """
        
        if os.path.exists(wtspath) and len(os.listdir(wtspath)) > 0:
            return 1
        elif os.path.exists(__PREFIX__ + "/weights/") and len(os.listdir(__PREFIX__+"/weights/")) == 0:
            os.rmdir(__PREFIX__+"/weights/")
        elif not os.path.exists(wtspath):
            os.mkdir(__PREFIX__ + "/weights/")
        
        with open(__PREFIX__+"/config/weights_download.json") as fp:
            json_file = json.load(fp)
        
        try:
            for net in ["pnet", "rnet", "onet"]:
                print("downloading {}.npy".format(net))
                url = 'https://drive.google.com/uc?id={}'.format(json_file[net])
                gdown.download(url, wtspath+"/{}.npy".format(net), quiet=False)       
        except Exception as exp:
            print("Error at weights download. ", exp)
            exit()
        
        return 0

    def detect(self, img):
        """detect [the main detection module used to run mtcnn inference on a 
        given image]

        Parameters
        ----------
        img : [str or numpy.ndarray or Image.Image]
            [the image to run mtcnn inference on]

        Returns
        -------
        [list]
            [returns bounding box coordinates of shape [num, 9] where 
            num -> number of face detections made
            Note: Bounding  box coordinates for each detection are 
                  given in (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            ]

        """

        if isinstance(img, str):
            if os.path.exists(img):
                img_name = os.path.basename(img)
                img = Image.open(img)
            else:
                raise FileNotFoundError("2",img)
        elif isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pass

        w, h = img.size
        min_length = min(h, w)
        min_detection_size = 12
        scale_factor = 0.709   #not sure why its .709
        scales = []
        m = min_detection_size/self.min_face_size
        min_length *= m
        factor_count = 0
        
        while min_length > min_detection_size:
            scales += [m * np.power(scale_factor,factor_count)]
            min_length *= scale_factor
            factor_count += 1

        ################## Stage 1 #############################

        bounding_boxes = []

        for s in scales:
            boxes = first_stage(img, s, self.pnet, self.nms_thresh[0])
            bounding_boxes.append(boxes)   
        #bounding_boxes has shape [n_scales, n_boxes, 9]
        
        #remove those scales for which bounding boxes were none
        bounding_boxes = [i for i in bounding_boxes if i is not None]

        #Add all the boxes for each scale 
        if len(bounding_boxes)==0:
            return bounding_boxes
        
        bounding_boxes = np.vstack(bounding_boxes)  # returns array of shape [n_boxes, 9]

        
        #------------------------- Stage 2 -------------------------------------
        
        img_box = get_image_boxes(bounding_boxes, img, size=24)   
        img_box = torch.tensor(img_box, dtype=torch.float32, requires_grad=False)

        probs, boxes = self.rnet(img_box)

        probs = probs.data.numpy() #Shape [boxes, 2]
        boxes = boxes.data.numpy() #Shape [boxes, 4]
        
        ind = np.where(probs[:, 1] >= self.conf_thresh[1])[0]

        bounding_boxes = bounding_boxes[ind]
        bounding_boxes[:, 4] = probs[ind, 1].reshape((-1,))
        boxes = boxes[ind]
        
        keep = nms(bounding_boxes, self.nms_thresh[1], mode="union")
        bounding_boxes = bounding_boxes[keep]
        boxes = boxes[keep]
        
        bounding_boxes = calibrate_boxes(bounding_boxes, boxes)
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
        
        #--------------------STAGE 3-------------------------------------------------

        img_box = get_image_boxes(bounding_boxes, img, size=48)
        
        if len(img_box) == 0:
            return [], []
        
        img_box = torch.tensor(img_box, dtype=torch.float32, requires_grad=False)
        probs, boxes, landmarks = self.onet(img_box)

        probs = probs.data.numpy()
        boxes = boxes.data.numpy()
        landmarks = landmarks.data.numpy()


        keep = np.where(probs[:,1] > self.conf_thresh[2])[0]

        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        boxes = boxes[keep]
        landmarks = landmarks[keep]
    
        bounding_boxes = calibrate_boxes(bounding_boxes, boxes)
    

        keep = nms(bounding_boxes, overlap_thresh=self.nms_thresh[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = convert_to_square(bounding_boxes)


        return bounding_boxes






    

