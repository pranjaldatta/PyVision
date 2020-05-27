# Multi Task Cascaded Convolutional Neural Network in PyTorch (MTCNN)

State of the art face and facial-landmark detection architecture.

## Paper

Read the paper [here](https://arxiv.org/pdf/1604.02878.pdf).

## Contributed By

- [Sashrika Surya](https://github.com/sashrika15)

- [Pranjal Datta](https://github.com/pranjaldatta)

## Tests

**All tests passing.**

To check, from PyVision root, run:

```
python tests/misc/mtcnn/mtcnn_test.py
```

## Usage

This Usage guide assumes that the PyVision repository has already been cloned. If not follow instructions given in PyVision repository root and clone the repository. Then follow the steps listed below:

```
from pyvision.misc.mtcnn import mtcnn
from PIL import Image
from pyvision.misc.mtcnn.utils.visualize import show_boxes

path = <path to image>

img = Image.open(path)

mtcnn = MTCNN()
boxes = mtcnn.detect(img) # returns bounding boxes

img = show_boxes(img, b)
img.show()
```

For a more detailed usage, check out [mtcnn_test.py](https://github.com/pranjaldatta/PyVision/blob/master/tests/misc/mtcnn/mtcnn_test.py)
