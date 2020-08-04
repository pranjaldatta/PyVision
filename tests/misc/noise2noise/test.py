from pyvision.misc.noise2noise.model import Noise2Noise
import cv2  
from PIL import Image
import numpy as np

def gaussian_noise(img):
    '''
    Add Gaussian noise in dataset
    Input: img of type PIL.Image
    Output: Noisy mage of type PIL.Image
    '''
    w,h = img.size
    c = len(img.getbands())

    sigma = np.random.uniform(20,50)
    gauss = np.random.normal(10,25,(h,w,c))
    noisy = np.array(img) + gauss
        
    #Values less than 0 become 0 and more than 255 become 255
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy)

    return img

n2n = Noise2Noise(noise="gaussian")

img_path = "/home/pranjal/Projects/clone/PyVision/tests/misc/noise2noise/test_images/test.jpg"
img = Image.open(img_path)
img = gaussian_noise(img)

img.show()
img.save("noised.png")
n2n.inference(img, show=False, save="denoised.png")