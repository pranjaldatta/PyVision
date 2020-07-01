#to run

from neural_style import Neural_Style

#provide the paths to the two images
style_img, content_img = ('images/style1.jpg', 'images/content1.jpg')

#if you do not wish to use gpu, pass use_gpu=False as a parameter, i.e., nst=Neural_Style(num_steps=300, use_gpu=False)
nst=Neural_Style(num_steps=300)

#call the function to run neural style transfer
output = nst.run_style_transfer(style_img, content_img)

nst.imshow(output)