import torch
import torch.nn as nn
import torch.nn.functional as F  

import numpy as np  
import cv2
import matplotlib.pyplot as plt  
from PIL import Image 


def letterbox_img(img, dims):
    """
    resize image keeping aspect ratio intact using padding
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = dims
    new_width = int(img_w * min(w/img_w, h/img_h))
    new_height = int(img_h * min(w/img_w, h/img_h))
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((dims[1], dims[0], 3), 128)

    canvas[(h-new_height)//2:(h-new_height)//2 + new_height, (w-new_width)//2:(w-new_width)//2 + new_width,  :] = img_resized

    return canvas

def prepare_img_cv2(img, dims):
    """
    prepare image for forward pass.

    returns a Tensor
    """
    # type check
    if not isinstance(img, np.ndarray):
        raise TypeError("expected <np.ndarray>. got <{}>".format(type(img)))
    
    img_dims = (img.shape[1], img.shape[0])
    _img = (letterbox_img(img, (dims, dims)))
    _img_new = _img[:,:,::-1].transpose((2,0,1)).copy()
    _img_new = torch.from_numpy(_img_new).float().div(255.0).unsqueeze(0)
    
    return _img_new, img, img_dims


def prepare_img_pil(img, dims):
    """
    prepares a PIL image for forward pass

    returns a Tensor
    """
    # type check
    if not isinstance(img, Image.Image):
        raise TypeError("expected <PIL.Image>. got <{}>".format(type(img)))
    
    original_img = img
    img = img.convert("RGB")
    img_dims = img.size
    img = img.resize(dims)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*dims, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, *dims)
    img = img.float().div(255.0)
    return (img, original_img, img_dims)


