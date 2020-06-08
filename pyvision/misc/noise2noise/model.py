import os
import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader
import gdown
import json
from .unet import Unet
from .dataset import NoisyDataset

root_dir = os.path.dirname(os.path.realpath(__file__))

class Noise2Noise:
    '''
    Noise2Noise class. 
    '''
    def __init__(self,data_path,noise,show=False):
        '''
        Initialise class
        '''
        print("Initialising Noise2Noise Model")
        self.show = show
        self.noise = noise
        self.data_path = data_path
    
        test_loader = self.load_dataset(data_path)

        if torch.cuda.is_available():
            self.map_location = 'cuda'
        else:
            self.map_location = 'cpu'
        
        self.check_weights()

        try:
            self.model = Unet(in_channels=3)
            self.load_model()
            
        except Exception as err:
            print("Error at {}".format(err))
            exit()

        self.save_path = self.save_path()    
        self.inference(test_loader)



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
        test_loader = DataLoader(dataset, batch_size=1)
        return test_loader

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


    def inference(self,test_loader):
        '''
        Inference of model
        Input: test_loader: Dataloader object
        '''
        source_imgs = []
        denoised_imgs = []   

        for source in list(test_loader):
            source_imgs.append(tvf.to_pil_image(torch.squeeze(source))) 
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
            
            

