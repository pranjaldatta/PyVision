import math
import numpy as np  
import torch
from .utils.utils import preprocess, nms
import cv2
from PIL import Image




def scale_boxes(probs, boxes, scale, thresh=.8):
    """
    A method that takes in the outputs of pnet, probabilities and 
    box cords for a scaled image and returns box cords for the 
    original image.

    Params:
    -> probs: probilities of a face for a given bbox; shape: [a,b]
    -> boxes: box coords for a given scaled image; shape" [1, 4, a, b]
    -> scale: a float denoting the scale factor of the image
    -> thresh: minimum confidence required for a facce to qualify

    Returns:
    -> returns a float numpy array of shape [num_boxes, 9] #9 because bbox + confidence + offset (4+1+4)
    """
    stride = 2
    cell_size = 12
    inds = np.where(probs > thresh)
    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [boxes[0, i, inds[0], inds[1]] for i in range(4)]  
    offsets = np.array([tx1, ty1, tx2, ty2])

    confidence = probs[inds[0], inds[1]]
    
    bboxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        confidence,
        offsets
        ])
     
    return bboxes.T


def first_stage(img, scale, pnet, nms_thresh):
    """
    A method that accepts a PIL Image, 
    runs it through pnet and does nms.

    Params:
    -> img: PIL image
    -> scale: a float that determines the scaling factor
    -> pnet: an instance of the pnet
    -> thresh: threshold below which facial probs are unacceptable

    Returns:
    -> numpy array of type float of shape [num_boxes, 9]
       which contain box cords for a givens scale, confidence,
       and offsets to actual size
    """
    
    orig_w, orig_h = img.size
    scaled_w, scaled_h = math.ceil(scale*orig_w), math.ceil(scale*orig_h)
    
    img = img.resize((scaled_w, scaled_h), Image.BILINEAR)
    img = preprocess(img)
    
    probs, boxes = pnet(img)
    
    
    probs = probs.data.numpy()[0,1,:,:] 
    boxes = boxes.data.numpy()

    bounding_boxes = scale_boxes(probs, boxes, scale)
    if len(bounding_boxes) == 0:
        return None

    selected_ids = nms(bounding_boxes[:,0:5], nms_thresh) #indices to be kept 
    return bounding_boxes[selected_ids]
   