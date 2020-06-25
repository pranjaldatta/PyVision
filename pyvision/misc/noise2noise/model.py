import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import gdown
from PIL import Image
import json
from .unet import Unet
from .dataset import NoisyDataset
from PIL import Image

root_dir = os.path.dirname(os.path.realpath(__file__))

class Noise2Noise:
    '''
    Noise2Noise class. 
    '''
    def __init__(self,data_path,noise,show=False,mode='inference'):
        '''
        Initialise class
        '''
        print("Initialising Noise2Noise Model")
        self.show = show
        self.noise = noise
        self.data_path = data_path
        self.mode = mode
        self.crop_size = 320

        if torch.cuda.is_available():
            self.map_location = 'cuda'
        else:
            self.map_location = 'cpu'
        

        try:
            self.model = Unet(in_channels=3)
            self.load_model()
            
        except Exception as err:
            print("Error at {}".format(err))
            exit()
        
        if mode=='inference':
            imgs = self.format_data(data_path)  
            self.save_path = self.save_path()  
            self.check_weights()
            self.inference(imgs)
        else:
            self.loss = nn.MSELoss()
            self.optim = Adam(self.model.parameters(),lr=1e-3)
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                factor=0.5, verbose=True)
            train_loader = self.load_dataset(data_path)
            self.train(train_loader)


    def format_data(self,data_path):
        imgs_path = []
        imgs = []
        for file in os.listdir(data_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                imgs_path.append( os.path.join(data_path,file))

        # Cropping Images
        for file in imgs_path:
            img = Image.open(file)
            w,h = img.size
            m = min(w,h)
            img = tvf.crop(img,0,0,m,m)
            img = tvf.resize(img,(self.crop_size, self.crop_size))
            imgs.append(img)
       
        return imgs

    
    def check_weights(self):

        if os.path.exists(root_dir + "/weights/n2n-{}.pt".format(self.noise)):
            print("Found weights")
        else:
            print("Downloading weights")
            self.download_weights()


    def download_weights(self):
        with open(root_dir+"/config/weights_download.json") as fp:
                json_file = json.load(fp)
                if not os.path.exists(root_dir+"/weights/"):
                    os.mkdir(root_dir+"/weights/")
                url = 'https://drive.google.com/uc?id={}'.format(json_file['n2n-{}.pt'.format(self.noise)])
                gdown.download(url, root_dir+"/weights/n2n-{}.pt".format(self.noise), quiet=False)
 
    
    def load_model(self):   
        ckpt_dir = root_dir + "/weights/n2n-{}.pt".format(self.noise)
        self.model.load_state_dict(torch.load(ckpt_dir, self.map_location))

    
    def load_dataset(self,img):
        dataset = NoisyDataset(img, self.noise, crop_size=256)
        train_loader = DataLoader(dataset, batch_size=5)
        return train_loader

    def save_path(self):
        '''
        Directory for output of model
        '''
        save_path = os.path.join(root_dir, 'Output')
        if not os.path.isdir(save_path):
            print("Making dir for denoised images")
            os.mkdir(save_path)
        print("Saving at {}".format(save_path))
        return save_path

    def crop_image(self,img):
        '''
        Crops the image to a square of size (crop_size, crop_size)
        Input: img of type PIL.Image
        Output: Cropped image of type PIL.Image
        '''

        w,h = img.size
        m = min(w,h)
        img = tvf.crop(img, 0,0,m,m)
        img = tvf.resize(img, (320, 320))
     
        return img

    def inference(self,imgs):
        '''
        Inference of model
        Input: path to directory containing images
        Input: test_loader: Dataloader object
        '''

        source_imgs = []
        denoised_imgs = []   

        for source in imgs:
            source_imgs.append(source)
            source = torch.unsqueeze(tvf.to_tensor(source),dim=0)
            output = self.model(source)
            denoised = tvf.to_pil_image(torch.squeeze(output))
            denoised_imgs.append(denoised)

        #Save images to directory
        for i in range(len(source_imgs)):
            source = source_imgs[i]
            denoised = denoised_imgs[i]
            source.save(os.path.join(self.save_path,'source_{}.png'.format(i+1)))
            denoised.save(os.path.join(self.save_path,'denoised_{}.png'.format(i+1)))
            if self.show==True:
                source.show()
                denoised.show()
        
        
    def train(self,train_loader):

        for epoch in range(2):
            print("Epoch {}/{}".format(epoch+1,2))
            for batch, (source,target) in enumerate(train_loader):
                denoised = self.model(source)
                loss = self.loss(denoised,target)
                print("Loss = ", loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()