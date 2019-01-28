import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),])
# get mnist
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# build model using Sequential (which is like tf.Keras)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10))
# define the loss
criterion = nn.CrossEntropyLoss()

# get data
images, labels = iter(trainloader).next()
images = images.view(images.shape[0], 784)

# forward pass
logits = model(images)
# calculate loss
loss = criterion(logits, labels)
print(loss)

"""
EXERCISE: Build a model that returns log-softmax as output and calculates the loss using negative log likelihood
loss.
"""
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1))
# define the loss
criterion = nn.NLLLoss()

# get data
images, labels = iter(trainloader).next()
images = images.view(images.shape[0], 784)

# forward pass
logits = model(images)
# calculate loss
loss = criterion(logits, labels)
print(loss)

# backward pass
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# to train our network, we need to use an optimizer
from torch import optim 

optimizer = optim.SGD(model.parameters(), lr=0.03)

# train the model!
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], 784)
        # zero out gradients in model before each pass
        optimizer.zero_grad()
        # forward pass
        logits = model.forward(images)
        loss = criterion(logits, labels)
        # backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# vis outputs
import helper

images, labels = next(iter(trainloader))

img = images[5].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)


