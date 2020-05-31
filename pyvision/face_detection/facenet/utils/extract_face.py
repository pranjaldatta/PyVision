import os  
import cv2 
from PIL import Image
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F  


def crop_and_tensorify(img, box, size=160, margin=0, save=None, show=True):
    """Extract face + margin from PIL Image given bounding box coordinates

    Arguments:
    -> img: PIL Image from which faces have to be extracted
    -> box: Bounding box coordinates in (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    -> size: size of the crop
    -> margin: around bounding boxes
    -> save: location to save the crop
    -> show: show the crops 

    Returns: 
    -> torch.Tensor: The face in a tensor format
    """
    if not isinstance(img, Image.Image):
        raise TypeError("PIL Image accepted. Got img of type: ", type(img))
    
    box = box[:4]

    margin = [
        margin * (box[2] - box[0]) / (size - margin),
        margin * (box[3] - box[1]) / (size - margin)
    ]

    box = [
        int(max(box[0] - margin[0]/2 , 0)),
        int(max(box[1] - margin[1]/2 , 0)),
        int(min(box[2] + margin[0]/2 , img.size[0])),
        int(min(box[3] + margin[1]/2 , img.size[1]))
    ]
 
    face = img.crop(box).resize((size, size), Image.BILINEAR)

    if save is not None:
        face.save(save+"/detection.png")
    if show:
        face.show()
    
    face = torch.tensor(np.float32(face))
    
    return face 

def prewhiten_func(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**5))
    y = (x - mean) / std_adj
    return y
        

def extract_face(mtcnn_module, img, prewhiten=True, conf_thresh=.6):
    """
    extract_face takes in a PIL or cv2 image or a path to an image.
    Runs MTCNN on the image to detect the face, crop the faces, convert 
    to tensor and return a tensor and the associated face confidences 

    Argument:
    -> img: PIL or cv2 Image. Can be a path to
    -> conf_thresh: Minimum confidence threshold for MTCNN

    Returns:
    -> face_tensors, props = cropped faces converted into tensors and their 
                              associated confidences repectively
    """
    
    if mtcnn_module is None:
        raise ValueError("mtcnn_module cannot be None")
    
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    
    # generating detections 
    detections = mtcnn_module.detect(img)

    # crop every face, convert to tensor
    faces_list = []
    for detection in detections:

        face = crop_and_tensorify(img, detection, show=False)
        if prewhiten:
            face = prewhiten_func(face)
        faces_list.append(face)

    faces_list = torch.stack(faces_list)    

    return faces_list # return face detections probs also

    
    
    
    
    
        

    