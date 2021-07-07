from pyvision.gans.deep_convolutional_gan import Deep_Conv_GAN

''' Initializing the DC_GAN module with the necessary paths '''
Deep_Conv_GAN.inference(Deep_Conv_GAN, set_weight_dir = 'dcgan-model.pth', set_gen_dir='result_img')
