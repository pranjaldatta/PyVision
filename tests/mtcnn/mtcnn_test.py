from mtcnn.detector import detector
from PIL import Image
import cv2
from mtcnn.utils.visualize import show_boxes, _show_boxes
from glob import glob


a = [glob("tests/mtcnn/images/*.{}".format(s)) for s in ["jpg", "jpeg", "png"]]
imgs = [i for ai in a for i in ai]
for img in imgs:
    img = Image.open(img)
    b = detector(img)
    try:
        img = show_boxes(img, b)
    except:
        img = _show_boxes(img, b)

    img.show()