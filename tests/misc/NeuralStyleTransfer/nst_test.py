#to run
from pyvision.misc.NeuralStyleTransfer import NeuralStyle

__PREFIX__ = "pyvision/misc/NeuralStyleTransfer/Examples/"
#provide the paths to the two images
style_img, content_img = (__PREFIX__+'images/style1.jpg', __PREFIX__+'images/content2.jpg')

#if you do not wish to use gpu, pass use_gpu=False as a parameter, i.e., nst=Neural_Style(num_steps=300, use_gpu=False)
nst = NeuralStyle(num_steps=300, retain_dims=False)

#call the function to run neural style transfer
output, time = nst.run_style_transfer(style_img, content_img)
print("time taken: ", time)

