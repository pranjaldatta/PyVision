# Noise2Noise
Noise2Noise is an image-denoising model which is trained on noisy data only. 
This implementation is based on the ICML 2018 paper. Check out the paper [here](https://arxiv.org/abs/1803.04189)!

### Summary
- This model works for additive gaussian noise and text removal only. It does not include poisson noise and Monte Carlo Rendering discussed in the paper.
- U-Net architecture is followed throughout the model. The original paper used a “RED30” network (Mao et al., 2016) for additive gaussian noise.
- The weights were made available by Joey Litalien's implementation [here](https://github.com/joeylitalien/noise2noise-pytorch).

### Test
To run test from PyVision root:
```
python tests/misc/noise2noise/n2n_test.py
```

### Usage
- The model setup is done through Noise2Noise class via pyvision.misc.noise2noise.model
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
    <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_gaussian/source_1.png" height=200 width=200></td>
    <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_gaussian/denoised_1.png" height=200 width=200></td>
  </tr>
  <tr>
  <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_gaussian/source_3.png" height=200 width=200></td>
  <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_gaussian/denoised_3.png" height=200 width=200></td>
  </tr>
 </table>

Text Overlay
<table>
  <tr>
    <td>Source Image</td>
     <td>Denoised Image</td>
   
  </tr>
  <tr>
    <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_text/source_1.png" height=200 width=200></td>
    <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_text/denoised_1.png" height=200 width=200></td>
  </tr>
  <tr>
  <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_text/source_3.png" height=200 width=200></td>
  <td><img src="https://github.com/pranjaldatta/PyVision/blob/sashrika-n2n/tests/misc/noise2noise/Output_text/denoised_3.png" height=200 width=200></td>
  </tr>
 </table>

