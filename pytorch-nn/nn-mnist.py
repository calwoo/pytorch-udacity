import numpy as np 
import torch
import matplotlib.pyplot as plt 
# we can get mnist via torchvision, which is basically a helpful starterpack of computer vision datasets
from torchvision import datasets, transforms

# make preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),])
# get mnist
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# DataLoader gives us an iterator to get our data, like tf.Datasets

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# dumb nn by hand
def sigmoid(x):
    return 1/(1+torch.exp(-x))
inputs = images.view((images.shape[0], 784))

W1 = torch.randn(784,256)
b1 = torch.randn(256)
W2 = torch.randn(256,10)
b2 = torch.randn(10)

h = sigmoid(torch.mm(inputs, W1) + b1)
out = torch.mm(h, W2) + b2

# softmax
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).unsqueeze(1)
probabilities = softmax(out)

"""
PyTorch makes building neural networks much easier with the nn module. This is sorta like tensorflow keras layers,
except in my opinion a lot more intuitive. A torch model just needs two parts:
    1) the architecture
    2) the forward pass
The backwards pass gets handled automatically by PyTorch!
"""
from torch import nn

# inheriting from nn.Module is mandatory for torch models
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        # define sigmoid/softmax activations
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = NN()

"""
A slightly cleaner way to build the neural net is to use the torch.nn.functional module. Compare:
"""
import torch.nn.functional as F 

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x

class AnotherNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# test it
model = AnotherNN()

images = images.view(images.shape[0], 784)
ps = model.forward(images[0].view(1,784))
img = images[0]

import helper
helper.view_classify(img.view(1,28,28), ps)
# so yeah, without training this is garbage