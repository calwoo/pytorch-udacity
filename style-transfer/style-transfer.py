import torch
import numpy as np 
import matplotlib.pyplot as plt 

from utils import *

# get the vgg19 model
vgg = get_vgg()
print(vgg)

content = load_image("imgs/octopus.jpg")
style = load_image("imgs/ben_passmore.jpg", shape=content.shape[-2:])
# show images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(numpy_image(content))
ax2.imshow(numpy_image(style))
plt.show()



