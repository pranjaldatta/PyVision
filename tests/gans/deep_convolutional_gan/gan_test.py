from pyvision.gans.deep_convolutional_gan import DeepConvGAN

''' Initializing the DC_GAN module with the necessary paths '''
DeepConvGAN.inference(DeepConvGAN, set_weight_dir = 'dcgan-model.pth', set_gen_dir='result_img')
