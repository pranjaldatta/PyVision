import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np  
import cv2  
import matplotlib.pyplot as plt

from .box_utils import iou


def load_classes(path):
    with open(path) as class_file:
        class_names = class_file.read().split("\n")[:-1]
    return class_names

def predict_transforms(preds, input_dims, anchors, n_classes, device='cpu'):

    batch_size = preds.size(0)
    stride = input_dims // preds.size(2)
    grid_size = input_dims // stride
    box_attrs = 5 + n_classes
    n_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    preds = preds.view(batch_size, box_attrs*n_anchors, grid_size*grid_size)
    preds = preds.transpose(1,2).contiguous()
    preds = preds.view(batch_size, grid_size*grid_size*n_anchors, box_attrs)

    preds[:,:,0] = torch.sigmoid(preds[:,:,0])
    preds[:,:,1] = torch.sigmoid(preds[:,:,1])
    preds[:,:,4] = torch.sigmoid(preds[:,:,4])

    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    
    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, n_anchors)
    x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0)

    preds[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    
    anchors = anchors.to(device)
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    preds[:,:,2:4] = torch.exp(preds[:,:,2:4])*anchors

    preds[:,:,5:5+n_classes] = torch.sigmoid(preds[:,:,5:5+n_classes])

    preds[:,:,:4] *= stride
    
    
    return preds


def _unique(t):

    t_numpy = t.cpu().numpy()
    t_np_unique = np.unique(t_numpy)
    t_unique = torch.from_numpy(t_np_unique)

    unique_tensor = torch.zeros(t_unique.shape) # error prone
    unique_tensor.copy_(t_unique)
    return unique_tensor






def postprocess(preds, device, confidence, n_classes, nms=True, nms_conf=0.5):
    """
    We perform confidence thresholding and nms suppression in this
    method
    """

    # confidence thresholding

    conf_mask = (preds[:,:,4] > confidence).float().unsqueeze(2)
    preds = preds*conf_mask
  

    # checks for non zero indices. If no non zero index remains 
    # shape of ind_nz will be (x, 0).  In that case we return 0
    try:
        ind_nz = torch.nonzero(preds[:,:,4]).transpose(0,1).contiguous()
        if ind_nz.size(1) == 0:
            raise Exception
    except:
        return 0
    

    # translate the coords from (center_x, center_y, height, width)
    # to (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    
    box_corners = torch.zeros_like(preds) #error prone
    box_corners[:,:,0] = (preds[:,:,0] - preds[:,:,2]/2)
    box_corners[:,:,1] = (preds[:,:,1] - preds[:,:,3]/2)
    box_corners[:,:,2] = (preds[:,:,0] + preds[:,:,2]/2)
    box_corners[:,:,3] = (preds[:,:,1] + preds[:,:,3]/2)
    preds[:,:,:4] = box_corners[:,:,:4]

    batch_size = preds.size(0)

    output = torch.zeros(1, preds.size(2) + 1) 
    write = False

    for index in range(batch_size):

        image_preds = preds[index]

        max_conf, max_conf_score = torch.max(image_preds[:,5:5+n_classes], 1) 
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        _seq = (image_preds[:,:5], max_conf, max_conf_score)
        image_preds = torch.cat(_seq, 1)

        non_zero_indices = (torch.nonzero(image_preds[:,4]))

        _image_preds = image_preds[non_zero_indices.squeeze(), :].view(-1,7)

        try:
            img_classes = _unique(_image_preds[:,-1])
            img_classes = img_classes.to(device)
        except:
            continue

        # now we do nms classwise 
        for _class in img_classes:

            cls_mask = _image_preds*(_image_preds[:,-1] == _class).float().unsqueeze(1)
            cls_mask_index = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = _image_preds[cls_mask_index].view(-1, 7)

            # sort the detections such that the entry with maximum objectness
            # score is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            num_dets = image_pred_class.size(0)

            if nms:

                # we run nms for each detection
                for i in range(num_dets):
                    
                    try:
                        ious = iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:], device)
                    except ValueError:
                        #print("ValueError: at iou calculation")
                        break
                    except IndexError:
                        #print("IndexError: at iou calculation")
                        break

                    # zero out all the entries whose iou value exceed the threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    # Remove the zero entries
                    non_zero_idx = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_idx].view(-1, 7)

                
                batch_inds = torch.zeros(image_pred_class.size(0), 1).fill_(index)
                batch_inds = batch_inds.to(device)
                _to_cat = (batch_inds, image_pred_class)

                if not write:
                    output = torch.cat(_to_cat, 1)
                    write = True
                else:
                    _outs = torch.cat(_to_cat, 1)
                    output = torch.cat((output, _outs))


    return output
