from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import time
import warnings

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# content loss
class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

# style loss
class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = self.gram_matrix(target_feature).detach()

        # gram matrix
        def gram_matrix(self, input):
            a, b, c, d = input.size()

            features = input.view(a * b, c * d)

            G = torch.mm(features, features.t())
            return G.div(a * b * c * d)

        def forward(self, input):
            G = self.gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
            
# for normalizing the input image
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.clone().detach().view(-1, 1, 1)
            self.std = std.clone().detach().view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std



class NeuralStyle(object):
    
    def __init__(self, content_layers=['conv_4'], style_layers= ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], 
        num_steps= 300, style_weight= 1000000, content_weight=1, use_gpu=False,
        retain_dims = True, downsample=True, save=None):

        """ 
        The Neural Style Transfer module. The class constructor is used to define some
        major parameters. The actual inference can be run with `run_style_transfer()`.

        Parameters:

        - content_layers (default: ['conv_4']): Layers to extract content features from

        - style_layers (default: ['conv_1', 'conv_2', 
            'conv_3', 'conv_4', 'conv_5']]): Layers to extracct style features from
        
        - num_steps (default: 300): Number of steps to run style transfer for

        - style_weight (default: 1000000): Weight given to style extracted from style 
                                            image
        
        - content_weight (default: 1): Weight given to cotnent extracted from content image

        - use_gpu (default: False): Run inference on gpu if available

        - retain_dims (default: True): Upsample output image to original dims since style transfer
                                       process downsamples to either 512/128. Warning: Upsampled image quality 
                                       may not be great
        
        - save (default: None): Path to save output image in. Should be of type path/to/*.jpg.
                                Doesnt save if None
        
        - downsample (default: True): If false, does not downsample to 512/128 i.e. style img is resized
                                      to size of content img and style transfer is run on original content img
                                      dimensions. VERY COMPUTATIONALLY intensive. Please ensure gpu is enabled.
        """
        if not isinstance(content_layers, list) or len(content_layers) < 1:
            raise ValueError("content_layers should be a list of len >= 1")
        if not isinstance(style_layers, list) or len(style_layers) < 1:
            raise ValueError("style_layers should be a list of len >= 1")
        if use_gpu and not torch.cuda.is_available():
            raise ValueError("use_gpu is True but cuda not available")
        if save is True or save is False:
            raise ValueError("save cannot be bool. Needs to be a path or None")
            
        if downsample == False:
           print("downsample = False does not set image size at 512 or 128. Performs style transfer on original content image size. Very computationally expensive. Please ensure gpu is enabled")
            

        
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.device = torch.device("cuda") if use_gpu else torch.device(("cpu"))
        self.imsize = 512 if (torch.cuda.is_available()) else 128 
        self.retain_dims = retain_dims
        self.save = save
        self.downsample = downsample

        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)


    # to convert a tensor to an image and then display it and save it in the desired path
    def imshow(self, tensor, title=None):
        unloader = transforms.ToPILImage()
        plt.ion()
        image = tensor.cpu().clone()
        image = image.squeeze(0)    
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        #provide the path for saving the output image
        image.save("output.png")

    # to resize images so that the aspect ratio of the content image remains the same. Then convert the images to tensors and 
    # create a white noise input image, and return the three tensors
    def image_loader(self, cnt, sty):

        if isinstance(cnt, str):
            if os.path.exists(cnt):
                img_name = os.path.basename(cnt)
                cntimg = Image.open(cnt)               
            else:
                raise FileNotFoundError("2",img)
        
        elif isinstance(cnt, np.ndarray):
            cntimg = Image.fromarray(cv2.cvtColor(cnt, cv2.COLOR_BGR2RGB))
        elif isinstance(img, Image.Image):
            pass

        if isinstance(sty, str):
            if os.path.exists(sty):
                img_name = os.path.basename(sty)
                styimg = Image.open(sty)               
            else:
                raise FileNotFoundError("2",sty)
        
        elif isinstance(sty, np.ndarray):
            styimg = Image.fromarray(cv2.cvtColor(sty, cv2.COLOR_BGR2RGB))
        elif isinstance(sty, Image.Image):
            pass
        
  
        w,h=cntimg.size
        ratio=w//h

        if self.downsample:
            l=max(w,h)
            s=min(w,h)

            if l>self.imsize:
            
                s=int(s/l*self.imsize)
                l=self.imsize
        
            if ratio==(s//l):
                styimg=styimg.resize((s,l))
                cntimg=cntimg.resize((s,l))
            elif ratio==(l//s):
                styimg=styimg.resize((l,s))
                cntimg=cntimg.resize((l,s))
        else:
            styimg = styimg.resize((w, h))
     
        loader = transforms.Compose([transforms.ToTensor()])
        
        content_image = loader(cntimg).unsqueeze(0)
        style_image = loader(styimg).unsqueeze(0)
        input_img = torch.randn(content_image.data.size(), device=self.device)
    
        content_image = content_image.to(self.device, torch.float)
        style_image = style_image.to(self.device, torch.float) 
        input_img = input_img.to(self.device, torch.float)
        input_dims = (w, h)

        return content_image, style_image, input_img, input_dims

    # to get the model and two lists of the style and content losses
    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                    style_img, content_img):

        content_layers=self.content_layers
        
        style_layers=self.style_layers
        
        cnn = copy.deepcopy(self.cnn)

        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    
    def run_style_transfer(self, style_img, content_img):

        num_steps = self.num_steps
        style_weight = self.style_weight
        content_weight = self.content_weight

        content_img, style_img, input_img, orig_dims = self.image_loader(content_img, style_img)

        #Run the style transfer.
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(self.cnn,
              self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img)
        # setting the optimizer according to the paper
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        run = [0]
        start_time = time.time()
        while run[0] <= num_steps:
            print("Step #{}".format(run[0]))
            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += (1/5)* sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("Step #{}:".format(run[0]))
                    print('Style Loss: {:4f} Content Loss: {:4f}'.format(
                          style_score.item(), content_score.item()))

                return style_score + content_score

            optimizer.step(closure)

        time_taken = time.time() - start_time

        input_img.data.clamp_(0, 1)

        unloader = transforms.ToPILImage()
        image = input_img.cpu().clone()
        image = image.squeeze(0)    
        image = unloader(image)

        if self.retain_dims:
            print("WARNING: retaining original dims can distort picture quality")
            image = image.resize(orig_dims, Image.BILINEAR)

        if self.save is not None:
            image.save(self.save)

        return image, time_taken
