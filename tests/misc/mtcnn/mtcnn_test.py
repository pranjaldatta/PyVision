from pyvision.misc.mtcnn import MTCNN
from pyvision.misc.mtcnn.utils.visualize import show_boxes, _show_boxes
from PIL import Image
import cv2
from glob import glob


a = [glob("tests/misc/mtcnn/images/*.{}".format(s)) for s in ["jpg", "jpeg", "png"]]
imgs = [i for ai in a for i in ai]

mtcnn = MTCNN()
for img in imgs:
    img = Image.open(img)
    b = mtcnn.detect(img)
    try:
        img = show_boxes(img, b)
    except:
        img = _show_boxes(img, b)

    img.show()