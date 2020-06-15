from pyvision.detection import detr
import time


imgs = ["tests/detection/detr/cars_test.jpg", "tests/detection/detr/zebra_test.jpg"]

print(detr.available_models()) # show available models

# testing on defualt detr-resnet50 
detr_object = detr.DETR(show=False) # make show True to see detections
print("Testing with detr-resnet50")
print("-"*50)
start_time = time.time()
for img in imgs:
    _, objs = detr_object.detect(img)
    print("No. of detections: ", len(objs))
    print("-"*50)

print("Total detection time: ", time.time() - start_time)
print("-"*50, end="\n\n")

# testing on detr-resnet101
detr_object = detr.DETR(model="detr-resnet101", show=False)
print("Testing with detr-resnet101")
print("-"*50)
start_time = time.time()
for img in imgs:
    _, objs = detr_object.detect(img)
    print("No. of detections: ", len(objs))
    print("-"*50)

print("Total detection time: ", time.time() - start_time)
print("-"*50)