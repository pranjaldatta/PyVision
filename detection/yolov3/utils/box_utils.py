import torch
import torch.nn as nn
import torch.nn.functional as F   

import numpy as np  
import cv2
import matplotlib.pyplot as plt  
import pickle as pkl
import random

def iou(box1, box2):
    """
    calculates iou between two boxes box1 and box2
    """
    b1x1, b1y1, b1x2, b1y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2x1, b2y1, b2x2, b2y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_x1 = torch.max(b1x1, b2x1)
    inter_y1 = torch.max(b1y1, b2y1)
    inter_x2 = torch.min(b1x2, b2x2)
    inter_y2 = torch.min(b1y2, b2y2)

    inter_shape = inter_x1.shape

    if torch.cuda.is_available():
        inter_area = torch.max(inter_x2-inter_x1+1.0, torch.zeros(inter_shape).cuda())*torch.max(inter_y2-inter_y1+1.0, torch.zeros(inter_shape).cuda())
    else:
        inter_area = torch.max(inter_x2-inter_x1+1.0, torch.zeros(inter_shape))*torch.max(inter_y2-inter_y1+1.0, torch.zeros(inter_shape))
    
    box1_area = (b1x2 - b1x1 + 1.0) * (b1y2 - b1y1 + 1.0)
    box2_area = (b2x2 - b2x1 + 1.0) * (b2y2 - b2y1 + 1.0)

    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def draw_box(pred, orig_img, cls, colors):
    """
    draw the predicted bounding boxes on a given image.
    designed for single images. 
    For multi batch support, supply singular image iteratively
    """

    coords1 = tuple(pred[1:3].int())
    coords2 = tuple(pred[3:5].int())
    label = "{0}".format(cls)
    color = random.choice(colors)
    cv2.rectangle(orig_img, coords1, coords2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    coords2 = coords1[0] + text_size[0] + 3, coords1[1] + text_size[1] + 4
    cv2.rectangle(orig_img, coords1, coords2, color, -1)
    cv2.putText(orig_img, label, (coords1[0], coords1[1]+text_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return orig_img    