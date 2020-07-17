# End-to-End Object Detection with Transformers (DEtection TRansformer)

DETR successfully uses Transformers in a conventional computer vision task such as detection. It reimagines the object detection pipeline and proposes an end-to-end pipeline. It views object detection as a **direct set prediction** problem. 

Check out [usage](#Usage) to start using DETR or check [summary](#Summary) for implementation details.

Do check out the [paper](https://scontent.fccu3-1.fna.fbcdn.net/v/t39.8562-6/101177000_245125840263462_1160672288488554496_n.pdf?_nc_cat=104&_nc_sid=ae5e01&_nc_ohc=sU420_xbxT8AX9LfbKI&_nc_ht=scontent.fccu3-1.fna&oh=455f6284084dfccdf0b9b39a878d290f&oe=5F0EB147) or visit the original GitHub [repository](https://github.com/facebookresearch/detr?fbclid=IwAR3Eqm_JaWigPZfi5Uk3Pdi24u_Y198n2twoTSvYnn22XmiBAN92lC3TgYA). (The visit is worth it! Not only they outline their approach in detail but also they demonstrate through a [colab notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb) how easy it is to make your own DETR in approx. 50 pytorch lines!)


Check out this standalone [notebook](https://github.com/pranjaldatta/PyVision/blob/master/demo/detection/detr/detr_demo.ipynb) to see how easily you can use YOLOv3 in 3-4 lines!
If the notebook link doesn't work, please look [here](https://nbviewer.jupyter.org/github/pranjaldatta/PyVision/blob/master/demo/detection/detr/detr_demo.ipynb) as a workaround.

## Summary

Currently, PyVision DETR supports the models listed below. The pretrained models were provided by the authors. More details can be accessed [here](https://github.com/facebookresearch/detr?fbclid=IwAR3Eqm_JaWigPZfi5Uk3Pdi24u_Y198n2twoTSvYnn22XmiBAN92lC3TgYA).

*Note:* Panoptic models are being added.

| Model| Train Dataset| Test Dataset | box AP | Available |
|--------|------------|------|---|----|
| DETR-Resnet50 (default) | COCO2017-val5k | COCO2017-val5k | 42.0 | Yes |
| DETR-Resnet101 | COCO2017-val5k | COCO2017-val5k | 43.5 | Yes

## Usage

For detailed documentation and parameters, refer to docstrings/source code.

**Brief Usage Summary:**

The model setup is done via the DETR class exposed in *PyVison.detection.detr* . All model related configuration ranging from model type to confidence thresholds can be set throught the class constructor.

Detection is done through the *detect()* method in the DETR class. Again, it offers some parameters for customisation that can override the general class configuration. Refer to source code docstrings for more details.

**Quick Start:**

- To use the default *DETR-Resnet50* model,

```
from pyvision.detection import detr

detr_obj = detr.DETR()

# time_taken is the total time taken to perform the detection
# result is the list of detections in a dict format {"scores": ..., "labels": ..., "coords": ...}

time_taken, result = detr_obj.detect(<img path or img in numpy format>)
```

- To use *DTER-Resnet101* model:

```
from pyvision.detection import detr

detr_obj = detr.DETR(model="detr-resnet101")

# time_taken is the total time taken to perform the detection
# result is the list of detections in a dict format {"scores": ..., "labels": ..., "coords": ...}

time_taken, result = detr_obj.detect(<img path or img in numpy format>)
```

- To list supported models,

```
from pyvision.detection import detr

print(detr.available_models())
```

- To run **tests**, from repo root, run the following command from terminal

```
$ python tests/detection/detr/detr_test.py
```

## Contributor

- [Pranjal Datta](https://github.com/pranjaldatta)
