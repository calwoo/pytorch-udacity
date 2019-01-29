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

# extract features following Gatys' paper on neural style transfer
def extract_features(image, model=vgg, layers=None):
    # if layers are passed in, use them, otherwise, use default layers from Gatys et al.
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', # content representation
            '28': 'conv5_1'}
    # get features as a dict
    features = {}
    for num, layer in model._modules.items():
        image = layer(image)
        if num in layers:
            features[layers[num]] = image
    return features

# gram matrix for getting style correlations
def gram_matrix(tensor):
    batch_size, d, h, w = tensor.size()
    tensor = tensor.reshape(d, h*w)
    tensor_tp = tensor.transpose(1,0)
    gram = torch.mm(tensor, tensor_tp)
    return gram

# get features of content and style
content_features = extract_features(content, vgg)
style_features = extract_features(style, vgg)
# get gram matrix of each layer of style representation
style_gram_matrices = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create target image. usually this is random noise, but we'll start it off as a copy of the content image
target = content.clone()
target.requires_grad = True

# weights for content and style losses
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}
alpha = 1
beta = 1e6

# optimizer
from torch import optim 
optimizer = optim.Adam([target], lr=0.003)

# training loop
epochs = 2000
for e in range(1, epochs+1):
    target_features = extract_features(target, vgg)
    # define losses
    content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"])**2)
    style_loss = 0
    for layer in style_weights:
        target_f = target_features[layer]
        target_gram = gram_matrix(target_f)
        style_gram = style_gram_matrices[layer]
        loss = torch.mean((target_gram - style_gram)**2)
        style_loss += loss * style_weights[layer]
    total_loss = alpha * content_loss + beta * style_loss

    # train
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if e % 400 == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()

# final display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))

