from  pyvision.segmentation.pspnet import PSPNet

m = PSPNet(model="pspnet-resnet50-ade20k")

m.inference("pyvision/segmentation/pspnet/examples/ade20k.jpg", save="ade20k")
