from PIL import Image
import numpy as np

def make_color_seg_map(seg_map_np, palette):
    color_img = Image.fromarray(seg_map_np.astype(np.uint8)).convert('P')
    color_img.putpalette(palette)
    return color_img