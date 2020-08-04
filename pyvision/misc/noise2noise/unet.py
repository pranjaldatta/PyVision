import torch
import torch.nn as nn 


class Unet(nn.Module):
    '''
    Unet architecture for n2n.
    No batch norm, dropout
    '''

    def __init__(self, in_channels=3, out_channels=3):
            """Initializes U-Net."""

            super(Unet, self).__init__()

            self._block1 = nn.Sequential(
                nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 48, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

            self._block2 = nn.Sequential(
                nn.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

            self._block3 = nn.Sequential(
                nn.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

            self._block4 = nn.Sequential(
                nn.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

            self._block5 = nn.Sequential(
                nn.Conv2d(144, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

            self._block6 = nn.Sequential(
                nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(0.1))


    def forward(self, x):

        #Encoder
        #print("X size = ", str(x.size()))
        pool1 = self._block1(x)
        #print(pool1.size())
        pool2 = self._block2(pool1)
        #print(pool2.size())
        pool3 = self._block2(pool2)
        #print(pool3.size())
        pool4 = self._block2(pool3)
        #print(pool4.size())
        pool5 = self._block2(pool4)
        #print(pool5.size())

        #Decoder
        upsample5 = self._block3(pool5)
        #print(upsample5.size())
        concat5 = torch.cat((upsample5, pool4), dim=1)
        #print(concat5.size())
        upsample4 = self._block4(concat5)
        #print(upsample4.size())
        concat4 = torch.cat((upsample4, pool3), dim=1)
        #print(concat4.size())
        upsample3 = self._block5(concat4)
        #print(upsample3.size())
        concat3 = torch.cat((upsample3, pool2), dim=1)
        #print(concat3.size())
        upsample2 = self._block5(concat3)
        #print(upsample2.size())
        concat2 = torch.cat((upsample2, pool1), dim=1)
        #print(concat2.size())
        upsample1 = self._block5(concat2)
        #print(upsample1.size())
        concat1 = torch.cat((upsample1, x), dim=1)
        #print(concat1.size())
        output = self._block6(concat1)
        #print(output.size())
        return output

    def summary(self):
        print('Unet summary: ')
        print(self._block1)
        print(self._block2)
        print(self._block3)
        print(self._block4)
        print(self._block5)
        print(self._block6)

        