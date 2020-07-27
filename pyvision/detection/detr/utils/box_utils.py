import torch
from torchvision.ops.boxes import box_area
import random
import cv2

clip = lambda x, x_min, x_max : x if x_min <= x <= x_max else (x_min if x < x_min else x_max)

def box_wh_to_xy(x):
    """
    Converts co-ordinates from (x, y, w, h) to 
    (x1, y1, x2, y2) format 
    """
    x, y, w, h = x.unbind(-1)
    
    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h

    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_xy_to_wh(x):
    """
    Converts co-ordinates from (x1, y1, x2, y2) to 
    (x, y, w, h) 
    """
    x1, y1, x2, y2 = x.unbind(-1)

    x = (x2 + x1)/2
    y = (y2 + y1)/2
    w = (x2 - x1)
    h = (y2 - y1)

    return torch.stack([x, y, w, h], dim=-1)

def iou(box1, box2):
    """
    Returns the iou between two boxes 
    """
    area1 = box_area(box1)
    area2 = box_area(box2)

    top_left = torch.max(box1[:, None, :2], box2[:, :2]) # remove None! very Irritating
    bottom_right = torch.min(box1[:, None, 2:], box2[:, 2:]) # remove None! Very Irritating

    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1 + area2 - inter #check this

    iou = inter / union

    return iou, union

def draw_box(orig_img, box, _cls, _cls_idx, colors, annotate):

    #img_w, img_h = orig_img.shape[0], orig_img.shape[1]
    #box[0:2]= [clip(x, 0.0, img_w) for x in box[0:2]]
    #box[1:4] = [clip(x, 0.0, img_h) for x in box[1:4]]

    coords1 = (int(box[0]), int(box[1]))   
    coords2 = (int(box[2]), int(box[3]))
    

    label = "{0}".format(_cls)
    color = colors[_cls_idx]
    cv2.rectangle(orig_img, coords1, coords2, color, 2)
    if annotate:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        coords2 = coords1[0] + text_size[0] + 3, coords1[1] + text_size[1] + 4
        cv2.rectangle(orig_img, coords1, coords2, color, -1)
        cv2.putText(orig_img, label, (coords1[0], coords1[1]+text_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    
    return orig_img    
    

def clamp(results, w_lim, h_lim):

    for idx in range(len(results)):
        box = results[idx]["coords"]
        box[0:2]= [clip(x, 0.0, w_lim) for x in box[0:2]]
        box[1:4] = [clip(x, 0.0, h_lim) for x in box[1:4]] 
        results[idx]["coords"] = box
    return results

