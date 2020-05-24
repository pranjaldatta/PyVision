# You Only Look Once v3 (YOLOv3)

YOLOv3 is a state of the art object detection algorithm.

Check out [usage](#Usage) to start using YOLOv3 in your project or check [summary](#Summary) for implementation details.

Do check out their [website](https://pjreddie.com/darknet/yolo/) or read the [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

## Summary

Currently, PyVision YOLOv3 supports the model listed below. The pretrained models were provided by the author. More details can be accessed [here](https://pjreddie.com/darknet/yolo/).

| Model     | Train Dataset| Test Dataset | mAP | FPS| Available |
--------|------------|------|---|----|-----|
| YOLOv3-416 (default) | COCO-trainval | test-dev | 55.3 | 35 | Yes |
| YOLOv3-tiny | COCO-trainval | test-dev | 33.1 | 220| Yes

## Usage

For detailed documentation and parameters, refer to docstrings/source code.

**Brief Usage Summary:**

The model setup is done via the YOLOv3 class exposed in *PyVison.detection.yolov3* . All model related configuration ranging from model type to confidence thresholds can be set throught the class constructor.

Detection is done through the *detect()* method in the YOLOv3 class. Again, it offers some parameters for customisation that can override the general class configuration. Refer to source code docstrings for more details.

**Quick Start:**

- To use the default *YOLOv3-416* model,

```
from PyVision.detection import yolov3

yolo = yolov3.YOLOv3()

# img is the images in array format with boxes drawn
# objs is the list of detections and box coordinates
imgs, objs = yolo.detect(<img path or img in numpy format>)
```

- To use *YOLOv3-tiny* model:

```
from PyVision.detection import yolov3

yolo = yolov3.YOLOv3(model="yolov3-tiny")

# img is the images in array format with boxes drawn
# objs is the list of detections and box coordinates
imgs, objs = yolo.detect(<img path or img in numpy format>)
```

- To list supported models,

```
from PyVision.detection import yolov3

print(yolov3.available_models())
```

- To run **tests**, from repo root, run the following command from terminal

```
$ python tests/detection/yolov3/yolo_test.py
```

## Contributor

- [Pranjal Datta](https://github.com/pranjaldatta)