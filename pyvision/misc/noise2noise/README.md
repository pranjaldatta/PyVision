# Noise2Noise: Learning Image Restoration without Clean Data

Noise2Noise is an image-denoising model which is trained on noisy data only.
This implementation is based on the ICML 2018 [paper](https://arxiv.org/abs/1803.04189) by Jaakko Lehtinen et al.

## Some Specific Details

### For denoising Gaussian noise

For Gaussian denoising, the model was trained with a *mean* of **10** and a *standard deviation* in the range [20, 50] (sampled randomly from an uniform distribution).

### For Text Removal

During the training of text removal model, random number of text units were added.

*For more details*, check out [dataset.py](https://github.com/pranjaldatta/PyVision/blob/master/pyvision/misc/noise2noise/dataset.py).

## Summary

- This model works for additive gaussian noise and text removal only. It does not include poisson noise and Monte Carlo Rendering discussed in the paper.
- U-Net architecture is followed throughout the model. The original paper used a “RED30” network (Mao et al., 2016) for additive gaussian noise.
- The weights were made available by Joey Litalien's implementation [here](https://github.com/joeylitalien/noise2noise-pytorch).
- For additive gaussian noise, sigma or the standard deviation is an important hyperparameter. If the **noise level is greater than thrice of sigma, the denoiser is unable to present a clear image**.
- The text overlay function works within a random integer range to add a random string to the image. The denoiser works better for small sized strings which cover less pixels.

### Test

To run test from PyVision root:

```python
python tests/misc/noise2noise/n2n_test.py
```

### Usage

- The model setup is done through Noise2Noise class via pyvision.misc.noise2noise.model
- The model is initialised with the noise type. For 'test' mode, a data_path is required which contains the path to test images. For 'inference' mode, a PIL image or the path to the image is required as input. The show parameter can be set to 'True' to display the images after denoising.
- The available noise types are: gaussian, text

```python
from pyvision.misc.noise2noise.model import Noise2Noise
from PIL import Image

n2n = Noise2Noise(noise="gaussian")

img_path = "Path to Image"
img = Image.open(img_path)

n2n.inference(img, show=False, save="Denoised.png")

```

### Example
Gaussian Noise:
<table>
  <tr>
    <td>Source Image</td>
     <td>Denoised Image</td>
   
  </tr>
  <tr>
    <td><img src="assets/gauss_1.png" height=200 width=200></td>
    <td><img src="assets/gdenoised_1.png" height=200 width=200></td>
  </tr>
  <tr>
  <td><img src="assets/gauss_3.png" height=200 width=200></td>
  <td><img src="assets/gdenoised_3.png" height=200 width=200></td>
  </tr>
 </table>

Text Overlay
<table>
  <tr>
    <td>Source Image</td>
     <td>Denoised Image</td>
   
  </tr>
  <tr>
    <td><img src="assets/text_1.png" height=200 width=200></td>
    <td><img src="assets/tdenoised_1.png" height=200 width=200></td>
  </tr>
  <tr>
  <td><img src="assets/text_3.png" height=200 width=200></td>
  <td><img src="assets/tdenoised_3.png" height=200 width=200></td>
  </tr>
 </table>

