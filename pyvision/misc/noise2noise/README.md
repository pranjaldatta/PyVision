# Noise2Noise

This is a PyTorch implementation of Noise2Noise. 

Paper: [noise2noise](https://arxiv.org/abs/1803.04189).

### Test
To run test from PyVision root:
```
python tests/misc/noise2noise/n2n_test.py
```
The noise type can be edited in n2n_test.py

### To run inference on pretrained model

```python

img_path = "path to noisy img"

n2n = Noise2Noise(noise="gaussian")

denoised_img = n2n.inference(img_path, show=False, save="denosied.png")
```

For more reference, please check test.py in project root. (To be deleted before final commit)

