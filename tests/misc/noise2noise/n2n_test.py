from pyvision.misc.noise2noise import Noise2Noise
import os

#data_path = <Path to directory containing images>
data_path = os.getcwd() + "/tests/misc/noise2noise/test_images"

#noise types: gaussian, text
n2n = Noise2Noise(noise='text',data_path=data_path,mode='test')
