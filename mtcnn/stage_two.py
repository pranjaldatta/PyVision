from .utils.utils import preprocess
import numpy as np
from PIL import Image
from .utils.visualize import show_boxes
from colorama import Fore

def get_image_boxes(bounding_boxes, img, size=24):

    """
    Cut out boxes from the image for rnet input
    """

    num_boxes = len(bounding_boxes)
    w, h = img.size
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(bounding_boxes, w, h)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((tmph[i], tmpw[i], 3), 'uint8')

        img_array = np.asarray(img, 'uint8')

        #Copies the values from img_array to empty img_box
        #x,ex,y,ey are the actual coords in the image
        try:
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] =\
                img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        except ValueError as ve:
            print("Value error at index {}".format(i))

        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = preprocess(img_box)

    return img_boxes


def pad(bboxes, width, height):
    """
    Output:
        dy, dx, edy, edx: Coordinates of cut boxes
        y, x, ey, ex: Coordinates of box in image
        h, w: Heights and widths of boxes.
    """
    
    #No idea why 1 is added and subtracted from w and h
    #e stands for end. So its (x,ex)

    x, y, ex, ey = [bboxes[:, i] for i in range(4)]
    w, h = ex - x + 1.0,  ey - y + 1.0
    num_boxes = bboxes.shape[0]
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    #For top left corner
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    #For bottom right corner 
    ind = np.where(ex > width - 1.0 )[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [r.astype('int32') for r in return_list]

    return return_list

