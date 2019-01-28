from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
# torch, and torchvision to get VGG19 model
import torch
from torchvision import transforms, models


def get_vgg(cuda=False):
    vgg = models.vgg19(pretrained=True)
    vgg_features = vgg.features
    """
    We freeze the parameters because in style transfer, we only want to use the VGG19 model as a feature
    extractor, not to train it via backpropagation. Instead, backprop will aim to minimize the loss function
    that compares the content/style representations of the target image with the output image
    """
    for param in vgg.parameters():
        param.requires_grad =False
    if cuda:
        device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
        vgg.to(device)

    return vgg



# image loader
def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (0.229, 0.224, 0.225))])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

# convert tensor image to numpy for display (matplotlib can't take pytorch)
def numpy_image(tensor_image):
    image = tensor_image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image