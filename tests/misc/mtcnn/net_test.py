from mtcnn.nets import ONet, PNet, RNet, FlattenTensorCustom
import torch
import numpy as np  
from colorama import Fore



pnet = PNet()
pnet.summary()

print("-"*50)

t = FlattenTensorCustom()
ar = np.random.rand(64, 3, 32, 32)
tensor = torch.FloatTensor(ar)
tensor = t(tensor)
if list(tensor.shape) == [64, 3*32*32]:
    pass
else:
    print(tensor.shape)
    print(Fore.RED+"ERROR: at FlattenTensorCustom Test"+Fore.RESET)
    exit()

print("-"*50) 

rnet = RNet()
rnet.summary()

print("-"*50) 

onet = ONet()
onet.summary()