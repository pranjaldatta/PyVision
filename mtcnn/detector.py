import numpy as np
from PIL import Image
import torch
from .nets import PNet, RNet, ONet
from .stage_one import first_stage
from .stage_two import get_image_boxes
from .utils.visualize import show_boxes
from .utils.utils import nms, convert_to_square, calibrate_boxes



def detector(image, min_face_size = 20.0, conf_thresh=[0.7, 0.7, 0.8], nms_thresh=[0.7, .7, .7]):
    """
    method that accepts an image and returns bounding boxes around faces

    Parameters:
    -> image (PIL.Image): Image in PIL format
    -> min_face_size (float): minimum size of face to look for
    -> conf_thresh (list): list of confidence thresholds for various parts
                           parts in the pipeine. (Size = 3)
    -> nms_thresh (list): list of overlap thresholds for nms (sizze = 3)
    """
    
    try:
        if not isinstance(image, Image.Image):
            raise TypeError
        if len(conf_thresh) != 3 or len(nms_thresh) != 3:
            raise AssertionError
    except AssertionError:
        print("ERROR: conf_thresh or nms_thresh of len :{},{} while expected size: 3".format(len(conf_thresh), len(nms_thresh)))
        exit()
    except TypeError:
        print("ERROR: Image type found:{}, expected: PIL.Image".format(type(image)))
        exit()

    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    
    w, h = image.size
    min_length = min(h, w)
    min_detection_size = 12
    scale_factor = 0.709   #not sure why its .709
    scales = []
    m = min_detection_size/min_face_size
    min_length *= m
    factor_count = 0
    
    while min_length > min_detection_size:
        scales += [m * np.power(scale_factor,factor_count)]
        min_length *= scale_factor
        factor_count += 1

    ################## Stage 1 #############################

    bounding_boxes = []

    for s in scales:
        boxes = first_stage(image, s, pnet, nms_thresh[0])
        bounding_boxes.append(boxes)   
    #bounding_boxes has shape [n_scales, n_boxes, 9]
    
    #remove those scales for which bounding boxes were none
    bounding_boxes = [i for i in bounding_boxes if i is not None]

    #Add all the boxes for each scale 
    if len(bounding_boxes)==0:
        return bounding_boxes
    
    bounding_boxes = np.vstack(bounding_boxes)  # returns array of shape [n_boxes, 9]

    
    #------------------------- Stage 2 -------------------------------------
    
    img_box = get_image_boxes(bounding_boxes,image,size=24)   
    img_box = torch.tensor(img_box, dtype=torch.float32, requires_grad=False)

    probs, boxes = rnet(img_box)

    probs = probs.data.numpy() #Shape [boxes, 2]
    boxes = boxes.data.numpy() #Shape [boxes, 4]
    
    ind = np.where(probs[:, 1] >= conf_thresh[1])[0]

    bounding_boxes = bounding_boxes[ind]
    bounding_boxes[:, 4] = probs[ind, 1].reshape((-1,))
    boxes = boxes[ind]
    
    keep = nms(bounding_boxes, nms_thresh[1], mode="union")
    bounding_boxes = bounding_boxes[keep]
    boxes = boxes[keep]
    
    bounding_boxes = calibrate_boxes(bounding_boxes, boxes)
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    
    #--------------------STAGE 3-------------------------------------------------

    img_box = get_image_boxes(bounding_boxes, image, size=48)
    
    if len(img_box) == 0:
        return [], []
    
    img_box = torch.tensor(img_box, dtype=torch.float32, requires_grad=False)
    probs, boxes, landmarks = onet(img_box)

    probs = probs.data.numpy()
    boxes = boxes.data.numpy()
    landmarks = landmarks.data.numpy()


    keep = np.where(probs[:,1] > conf_thresh[2])[0]

    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    boxes = boxes[keep]
    landmarks = landmarks[keep]
   
    bounding_boxes = calibrate_boxes(bounding_boxes, boxes)
   

    keep = nms(bounding_boxes, overlap_thresh=nms_thresh[2], mode="min")
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = convert_to_square(bounding_boxes)


    return bounding_boxes

