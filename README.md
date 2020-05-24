
# PyVision

Ready-to-use implementations of some of the most common computer vision algorithms.

In PyTorch only!

## Currently available architectures

- **Multi Task Cascaded Convolutional Neural Network (MTCNN)** : A SOTA face and facial-landmark detection architecture. Check out [this](https://github.com/pranjaldatta/PyVision/blob/master/mtcnn/README.md) out for more details.

- **YOLOv3:** The SOTA object detection algorithm. For more details, read the [docs](https://github.com/pranjaldatta/PyVision/blob/master/detection/yolov3/readme.md)

## Installation 

Run the following code to only clone the master branch which contains working code

```
git clone https://github.com/pranjaldatta/PyVision.git --single-branch --branch master
```

## Contributing

- When contributing, the complete source files + tests should be pushed through a branch by the name of the contributer.

- tests should be in the PyVision/tests/\<architecture name\>/ folder.

- If weights are heavy, please ensure they are downloaded **only** on an as-you-need basis. (Maybe host it somewhere like AWS S3)

## ToDo

- [ ] Populate with more architectures (obviously)

- [x] ~~Come up with an efficient way to make the repository minimal i.e. assets (like weights) will only be downloaded on as-you-need basis.~~ All weights are hosted on SRM-MIC Google drive and downloaded using gdown

- [x] ~~Come up with an efficient way to ensure that heavy architecture specific dependecies are installed only when required.~~ All heavy assets are installed only when model is being used.

## Note

Currently, its working only in pre-configured conda environment with all dependencies installed.
