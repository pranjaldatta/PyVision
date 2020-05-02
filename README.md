
# PyVision

Ready-to-use implementations of some of the most common computer vision algorithms.
In PyTorch only!

## Currently available architectures

- **Multi Task Cascaded Convolutional Neural Network (MTCNN)** : A SOTA face and facial-landmark detection architecture. Check out this for more details.

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

- [ ] Come up with an efficient way to make the repository minimal i.e. assets (like weights) will only be downloaded on as-you-need basis.

- [ ] Come up with an efficient way to ensure that heavy architecture specific dependecies are installed only when required.

## Note

Currently, its working only in pre-configured conda environment with all dependencies installed.
