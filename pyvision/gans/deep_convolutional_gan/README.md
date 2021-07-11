# Deep Convolutional GAN
This is an implementation of the research paper <a href = "https://arxiv.org/abs/1511.06434.pdf">"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"</a> written by Alec Radford, Luke Metz, Soumith Chintala.

Check out this <a href = "https://colab.research.google.com/drive/1rz1NZK0m0b5xxcLrgtEOpvIsl3aEfUtJ?usp=sharing">notebook</a> and run the DC_GAN inferences in just 3 lines.

## Dependencies
- torch==1.8.0
- torchvision==0.9.0
- numpy==1.20.3
- matplotlib==3.3.4
- IPython==7.23.1
- gdown==3.13.0

## Dataset
The original paper had used three datasets for training the DCGAN namely - *Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset*. However due to computational and other limitations, we have used <a href = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">Large-scale CelebFaces Attributes (CelebA) Dataset</a>.

### Guidelines to download, setup and use the dataset
The CelebA dataset may be downloaded <a href = "https://drive.google.com/file/d/1yW6QkWcd6sWYB2rw9d-A36woiXVLTpny/view?usp=sharing">here</a> as a file named *img_align_celeba.zip*. 

**Please write the following commands on your terminal to extract the file in the proper directory**
```
$ mkdir celeba
$ unzip </path/to/img_align_celeba.zip> -d </path/to/celeba>
```
The resulting directory structure should be:
```
/path/to/celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
```
<br>

**Note**: You may use any other dataset of your choice. However, please ensure that the directory structure remains the same for the code to be compatible with it.

## Quick Start
- Incase you want to use some other dataset to train the DCGAN, please initialize the DCGAN module with your desired dataset path and train as:

```python
from dcgan import DCGAN

dc_gan = DCGAN(data = <path/to/dataset>)
img_list, G_losses, D_losses = dc_gan.train(<path/to/save/model>)
```

- Incase you have either no GPU (0) or more than 1 GPU on your machine, consider changing the ngpu parameter while initializing the DCGAN module with your desired dataset path and train as:


```python
from dcgan import DCGAN

dc_gan = DCGAN(data = <path/to/dataset>, ngpu = <number of GPUs available>)
img_list, G_losses, D_losses = dc_gan.train(<path/to/save/model>)
```

**Note**: Is is advisable to use a GPU for training because training the DCGAN is computationally very expensive.

- To get the inferences directly with our pre-trained model please initialize the DeepConvGAN with the desired path to the model and get the inferences as:

```python
from model import DeepConvGAN

DeepConvGAN.inference(DeepConvGAN, set_weight_dir='dcgan-model.pth' , set_gen_dir='<path/to/save/inferences>')
```

## Tests
To run tests from PyVision root, run,

    $ python tests/gans/deep_convolutional_gan/gan_test.py

## Results from implementation
- Plot to see how D and G’s losses changed during training

<img src = "/pyvision/gans/deep_convolutional_gan/results/losses.png">

- Batches of fake data from G

<img src = "/pyvision/gans/deep_convolutional_gan/results/result.png" height = 350px width = 350px> &nbsp; &nbsp; <img src = "/pyvision/gans/deep_convolutional_gan/results/result2.png" height = 350px width = 350px>

Check out the documentation <a href = "https://github.com/indiradutta/PyVision/blob/master/pyvision/gans/deep_convolutional_gan/docs/documentation.md">here</a>.

### Citation
``` 
@inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {December},
 year = {2015} 
}
```

## Contributed by:
- <a href = "https://github.com/indiradutta">Indira Dutta</a>
- <a href = "https://github.com/srijarkoroy">Srijarko Roy</a>
