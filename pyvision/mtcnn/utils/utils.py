import numpy as np 
from PIL import Image
import torch

def nms(boxes, overlap_thresh=.5, mode='union'):
    """
    An utility function that performs nms over the bounding box

    Params:
    -> boxes: the bounding box proposals
    -> overlap_thresh: maximum permissible overlap ratio
    -> mode: default - union (IoU)

    Output:
    -> bounding box list with overlapping boxes removed
    """

    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2, confidence = [boxes[:, i] for i in range(5)]

    areas = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    selected = []
    ids_sorted = np.argsort(confidence)

    while len(ids_sorted) > 0:
        """
        we loop through the sorted ids. 
        1. select the last id
        2. compare the chosen bbox IoU with all the others
        3. del the ones above the threshold.
        4. return selected ids
        """        

        last_idx = len(ids_sorted) - 1
        idx = ids_sorted[last_idx]
        selected.append(idx)
        

        xi1 = np.maximum(x1[idx], x1[ids_sorted[:last_idx]])
        yi1 = np.maximum(y1[idx], y1[ids_sorted[:last_idx]])

        xi2 = np.minimum(x2[idx], x2[ids_sorted[:last_idx]])
        yi2 = np.minimum(y2[idx], y2[ids_sorted[:last_idx]])

        inter_h = np.maximum(0.0, (yi2 - yi1 + 1.0))
        inter_w = np.maximum(0.0, (xi2 - xi1 + 1.0))
        inter_area = inter_h*inter_w

        if mode == "union":
            overlap = inter_area/(areas[idx] + areas[ids_sorted[:last_idx]] - inter_area)
        elif mode == "min":
            overlap = inter_area/np.minimum(areas[idx], areas[ids_sorted[:last_idx]])

        to_del = np.concatenate([[last_idx], np.where(overlap > overlap_thresh)[0]])
        ids_sorted = np.delete(ids_sorted, to_del)

    #print("nms complete. returning {}/{} boxes".format(len(selected), len(boxes)))
    return selected



def preprocess(img):
    """
    A utiity function that takes a numpy image array or PIL
    Image and returns a tensor
    
    Input: 
        -> img: input image in array or PIL format
    Output:
        -> tensor    
    """
    if isinstance(img, Image.Image):
        img = np.asarray(img, 'float')
    img = torch.tensor(img, dtype=torch.float32, requires_grad=False)
    img = img.permute(2,0,1)
    img = torch.unsqueeze(img, 0)
    img = (img - 127.5)*0.0078125  #normalize
    return img
    
def convert_to_square(bbox):
    """
    Convert bounding boxes to square shape
    
    """

    square = np.zeros((bbox.shape))

    x1, y1, x2, y2 = [bbox[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)

    square[:,0] = x1 + w*0.5 - max_side*0.5
    square[:,1] = y1 + h*0.5 - max_side*0.5
    square[:, 2] = square[:, 0] + max_side - 1.0
    square[:, 3] = square[:, 1] + max_side - 1.0

    return square

def calibrate_boxes(boxes, offsets):
    '''
    offset the original bounding boxes by an amount as predicted by the 
    rnet.

    Arguments:
    -> boxes: original bounding box list (shape: [n, 9])
    -> offsets: output of the rnet (shape [n, 4])

    Returns:
    -> numpy array of shape [n, 5]
    '''
    
    x1, y1, x2, y2 = [boxes[:,i] for i in range(4)]

    width = (x2 - x1 + 1.0)
    height = (y2 - y1 + 1.0)
    
    height = np.reshape(height, (-1, 1))
    width = np.reshape(width, (-1, 1))

    tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    t = [x1, y1, x2, y2, tx1, ty1, tx2, ty2]
    t = list(map(lambda x: np.reshape(x,(-1, 1)), t))
    x1, y1, x2, y2, tx1, ty1, tx2, ty2 = t[:]
    
    """
    it was supposed to be x1t = x1+tx1*width but that was providing negative indices so swapped
    tx1 and tx2
    """
    x1t = x1 + tx2*width
    y1t = y1 + ty1*height
    x2t = x2 + tx1*width
    y2t = y2 + ty2*height

    t = [x1t, y1t, x2t, y2t]
 
    t = list(map(lambda x: np.reshape(x, (-1,)), t))
    for i in range(4):
        boxes[:,i] = t[i]
    return boxes
