from pyvision.detection import yolov3
print(yolov3.available_models())
yolo = yolov3.YOLOv3(model="yolov3-tiny", device="gpu")
_, objs = yolo.detect("F:\clone\PyVision\\tests\detection\yolov3\zebra_test.jpg")
print(objs)
