from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

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
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std



class Neural_Style:
    
    def __init__(self, content_layers_default=['conv_4'], style_layers_default= ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], num_steps= 300, style_weight= 1000000, content_weight=1, use_gpu=True):
    
        self.content_layers_default = content_layers_default
        self.style_layers_default = style_layers_default
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.device = if torch.device("cuda" if (torch.cuda.is_available()) and use_gpu else "cpu")
        self.imsize = 512 if (torch.cuda.is_available()) else 128 

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
        styimg = Image.open(sty)
        cntimg= Image.open(cnt)
        

        w,h=cntimg.size
        ratio=w//h

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
         
        loader = transforms.Compose([transforms.ToTensor()])
        
        content_image = loader(cntimg).unsqueeze(0)
        style_image = loader(styimg).unsqueeze(0)
        input_img = torch.randn(content_image.data.size(), device=self.device)
    
        return content_image.to(self.device, torch.float),style_image.to(self.device, torch.float),input_img.to(self.device, torch.float)

    # to get the model and two lists of the style and content losses
    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                    style_img, content_img):

        content_layers=self.content_layers_default
        
        style_layers=self.style_layers_default
        
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

        content_img, style_img, input_img = self.image_loader(content_img, style_img)

        #Run the style transfer.
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(self.cnn,
              self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img)
        # setting the optimizer according to the paper
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

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
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                          style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

          
        input_img.data.clamp_(0, 1)
        
        return input_img
