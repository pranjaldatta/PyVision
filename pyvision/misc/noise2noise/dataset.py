import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters


class NoisyDataset(Dataset):
    '''
    Loads dataset.
    NoisyDataset inherits from an abstract class representing Dataset
    '''

    def __init__(self, data_dir, noise, crop_size):
        '''
        Initialise dataset
        '''
        self.data_dir = data_dir
        self.imgs = []
        self.crop_size = 320
        self.noise = noise

        for file in os.listdir(data_dir):
            if file.endswith(".jpg"):
                self.imgs.append( os.path.join(data_dir,file))
    
    def gaussian_noise(self,img):
        '''
        Add Gaussian noise in dataset
        Input: img of type PIL.Image
        Output: Noisy image of type PIL.Image
        '''
        w,h = img.size
        c = len(img.getbands())

        sigma = np.random.uniform(20,50)
        gauss = np.random.normal(10,sigma,(h,w,c))
        noisy = np.array(img) + gauss
        
        #Values less than 0 become 0 and more than 255 become 255
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy)

        return img

    
    def add_text(self,img):
        '''
        Add random string of text to images
        Input: img of type PIL.Image
        Output: Noisy image of type PIL.Image
        '''
        w,h = img.size
        c = len(img.getbands())
        im = img.copy()
        draw = ImageDraw.Draw(im)
        for i in range(random.randint(5,15)):
            font_type = ImageFont.truetype(font='Arial.ttf',size=np.random.randint(10,20))
            len_text = np.random.randint(4,20)
            text = ''.join(random.choice(ascii_letters) for i in range(len_text))
            x = np.random.randint(0,w)
            y = np.random.randint(0,h)
            col = tuple(np.random.randint(0,255,c))
            draw.text((x,y),text,fill=col,font=font_type)

        return im
    
    
    def crop_image(self,img):
        '''
        Crops the image to a square of size (crop_size, crop_size)
        Input: img of type PIL.Image
        Output: Cropped image of type PIL.Image
        '''
        
        w,h = img.size
        m = min(w,h)
        img = tvf.crop(img, 0,0,m,m)
        img = tvf.resize(img, (self.crop_size, self.crop_size))
     
        return img
    

    def __len__(self):
        '''
        Returns length of dataset
        '''
        return len(self.imgs)


    def __getitem__(self,index):
        '''
        Compiles dataset
        '''
        
        img =  Image.open(self.imgs[index]).convert('RGB')
        resized_img = self.crop_image(img)

        if self.noise == 'text':
            source = tvf.to_tensor(self.add_text(resized_img))
            target = tvf.to_tensor(self.add_text(resized_img))
        else:
            source = tvf.to_tensor(self.gaussian_noise(resized_img))
            target = tvf.to_tensor(self.gaussian_noise(resized_img))
        
        return source,target


 