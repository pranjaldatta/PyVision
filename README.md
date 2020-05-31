
# PyVision

Ready-to-use implementations of some of the most common computer vision algorithms.

In PyTorch only!

## Currently available architectures

- **Multi Task Cascaded Convolutional Neural Network (MTCNN)** : A SOTA face and facial-landmark detection architecture. Check out [this](https://github.com/pranjaldatta/PyVision/blob/master/mtcnn/README.md) out for more details.

- **YOLOv3:** The SOTA object detection algorithm. For more details, read the [docs](https://github.com/pranjaldatta/PyVision/blob/master/detection/yolov3/readme.md).

- **FaceNet: A Unified Embedding for Face Recognition and Clustering**: One of the most popular architectures used for facial recognition. For more details, check [here](https://github.com/pranjaldatta/PyVision/tree/master/pyvision/face_detection/facenet).

For full list of architectures that has been ported or are **in the process** of being ported, check [here](https://github.com/pranjaldatta/PyVision/blob/master/docs/developing.md).

## Installation

1. Run the code in your terminal to clone the master branch which contains the working code

```
$ git clone https://github.com/pranjaldatta/PyVision.git --single-branch --branch master
```

2. Then, go to the repository root by pasting the command given below into your terminal

```
$ cd PyVision
```

3. Run the following command in the terminal to install PyVision into the current virtual or conda environment

```
$ pip install .
```

4. You are good to go!.

## Contributing

For contribution guidelines, please look [here](https://github.com/pranjaldatta/PyVision/tree/master/docs/contributing.md).  Contributions are always welcome!

## ToDo

- [ ] Populate with more architectures (obviously)

- [x] ~~Come up with an efficient way to make the repository minimal i.e. assets (like weights) will only be downloaded on as-you-need basis.~~ All weights are hosted on SRM-MIC Google drive and downloaded using gdown

- [x] ~~Come up with an efficient way to ensure that heavy architecture specific dependecies are installed only when required.~~ All heavy assets are installed only when model is being used.

## Note

Currently, its working only in pre-configured conda environment with all dependencies installed.
