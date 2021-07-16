## Model Components
The DCGAN Architecture has the following components:

- The Generator uses fractional-strided convolutions followed by batch normalisation and ReLU activation for all layers except for the last that uses tanh activation.
- The Discriminator uses strided convolutions followed by batch normalisation and LeakyReLU activation for all layers except for a single sigmoid output.
<img src="https://miro.medium.com/max/846/1*rdXKdyfNjorzP10ZA3yNmQ.png" >

## Parameters

Parameter |  &nbsp;&nbsp;&nbsp;&nbsp; Value &nbsp;&nbsp;&nbsp;&nbsp; |
:------------: | :---: |
batch_size | 128 |
image_size | 64 |
nc | 3 |
nz | 100 |
ngf | 64 |
ndf | 64 |
num_epochs | 5 |
lr | 0.0002 |
beta1 | 0.5 |
ngpu | 1 |

## Result Documentation
After running *DCGAN* on the CelebA Dataset for 5 epochs on GPU (computationally very expensive) we got the following output images along with the Generator and Discriminator losses.

## Batch of images from the Generator after 5 epochs 
<img src="/pyvision/gans/deep_convolutional_gan/results//result2.png">

## Losses after each epoch
No. of Epochs | Generator Loss | Discriminator Loss |
:------------: | :------------: | :------------: |
1 | 0.7894 | 1.0838 |
2 | 0.7277 | 1.0489 |
3 | 0.7796 | 0.9256 |
4 | 0.6330 | 1.1345 |
5 | 0.7519 | 1.0138 |

## Plot for Generator Loss and Discriminator Loss w.r.t number of iterations
<img src="/pyvision/gans/deep_convolutional_gan/results/losses.png">

