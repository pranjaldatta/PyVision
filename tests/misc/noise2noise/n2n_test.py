from pyvision.misc.noise2noise import Noise2Noise
from glob import glob
from PIL import Image
import os
import sys

#data_path = <Path to directory containing images>
data_path = "/Users/sashrikasurya/Documents/PyVision/tests/misc/noise2noise/test_images"

#noise types: gaussian, text
n2n = Noise2Noise(data_path,noise='gaussian')
