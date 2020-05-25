from pyvision.detection import yolov3
import time


imgs = ["tests/detection/yolov3/cars_test.jpg", "tests/detection/yolov3/zebra_test.jpg"]

print(yolov3.available_models()) # show available models

# testing on defualt yolov3-416 
yolo = yolov3.YOLOv3(show=False) # make show True to see detections
print("Testing with yolov3-416")
print("-"*50)
start_time = time.time()
for img in imgs:
    _, objs = yolo.detect(img)
    print("No. of detections: ", len(objs))
    print("-"*50)

print("Total detection time: ", time.time() - start_time)
print("-"*50, end="\n\n")

# testing on yolov3-tiny
yolo = yolov3.YOLOv3(model="yolov3-tiny", show=False)
print("Testing with yolov3-tiny")
print("-"*50)
start_time = time.time()
for img in imgs:
    _, objs = yolo.detect(img)
    print("No. of detections: ", len(objs))
    print("-"*50)

print("Total detection time: ", time.time() - start_time)
print("-"*50)