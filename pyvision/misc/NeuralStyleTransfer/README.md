## Neural Style Transfer: An implementation of the paper "A Neural Algorithm of Artistic Style".

Link to the paper: https://arxiv.org/pdf/1508.06576.pdf

The idea is to extract the _content_ from one image, the 'content image', and the _style_ or _texture_ from another image, the 'style image', to get a single output which has a combination of the two.

## Requirements:

* matplotlib==3.1.2
* numpy==1.18.4
* Pillow==7.1.2
* torch==1.5.0
* torchvision==0.6.0

## Steps required to run:

from pyvision.misc.NeuralStyleTransfer.neural_style import Neural_Style

style_img, content_img = ('path_to_style_image', 'path_to_content_image')

nst=Neural_Style(num_steps=300)

output = nst.run_style_transfer(style_img, content_img)

nst.imshow(output)

