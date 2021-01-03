import cv2 
from PIL import Image
from pyvision.detection import efficientdet

model = efficientdet.EfficientDet("coco", thresh=0.95)

img1 = cv2.imread("tests/detection/effdet/2.jpg")
img2 = cv2.imread("tests/detection/effdet/3.jpg")

imgs = [img1, img2]

for img in imgs:
    img = cv2.resize(img, (416, 416))
    res = model.detect(img)
    cv2.imshow("Frame", res[0])
    if cv2.waitKey() == ord('q'):
        continue