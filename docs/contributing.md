# Contribution Guidelines

PyVision is meant to be a collection of all major (popular or otherwise) computer vision architectures made available in an easy-to-use, extensible format so that the net number of lines required to use the architecture, whether for training or inference purposes are reduced to **three/four** lines.

The objective of such an effort is two fold:

- To develop a library for our own use, that simplifies computer vision architecture use so that developers can focus on the project they are working on and not bother about the nuances and headaches of complex implementations.

- To learn the nuances and deal with the headaches of complex architecture implementations and hopefully, become better engineers!


## Why Contribute?

- Learn the details of seemingly complex architectures.

- Learn the nuances of implementation.

- Help make computer vision easier and more approachable!

## How to Contribute? 

The following steps detail roughly the contribution workflow

1. Decide on an architecture you want to implement! Once decided, **open an issue** at [issues](https://github.com/pranjaldatta/PyVision/issues). Be sure to classify the architecture under a given category. *For Example*, YOLOv3 falls under the category of *detection*. If unsure, ask in the issue. 

2. Once you are sure no one else is working on the given architecture, clone the master repository.

3. Once in local repository root, create a branch with your name and architecture you are working on. *An example branch name:* **pranjal-yolov3**.
To create a new branch, run the following command in the local repo root from your terminal,

```
$ git checkout -b <branch-name>
```

4. Code!

5. **Important**: The most critical issue here is regarding the model **weights**. The weights of a given model, **does not come** pre-loaded with the repository. This is done because, 
    - To reduce the repository size (obviously).
    - GitHub doesn't allow hosting of files of sizes more than 100 MB.
    - Also, making model weights available **lazily** is more efficient as people are downloading **only** those weights that they are using

    So, whats the solution? For the detailed process check [this](https://github.com/pranjaldatta/PyVision/blob/master/docs/weights.md).

    **TL;DR**:
    - Provide the maintainer with links to the downloadable weights in the issue. The maintainer will download the weights and upload it to SRM-MIC's Google Drive.
    
    - Will provide the **file id** to the contributor.

    - Download the weights in a **lazy** manner only when the **model is being initialized** using **gdown**

    - Check YOLOv3's download_weights() method for reference.

6. Add tests! The tests should be self-contained and folder structure should be maintained in the [tests folder](https://github.com/pranjaldatta/PyVision/blob/master/tests) as it is maintained in the repo root. 

7. **Very Important**: Add docs! Add docstrings to classes, functions. **How to use** along with example code is a must. Try to cover everything in documentation whether as markup or in source code.

8. Once you are done, push the branch **referring** the issue! Resolve any problems/inconsistencies brought to your notice and wait for the merge!
