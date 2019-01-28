import torch

def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7) # set a random seed
features = torch.randn((1,5))
# it is awesome how close this is to numpy
weights = torch.randn_like(features)
bias = torch.randn((1,1))

# output of a simple neuron
y = activation(torch.sum(features * weights) + bias)
# could also do this via matrix mult
y = activation(torch.mm(features, weights.reshape((5,1))) + bias)
print(y)

# now lets do a simple multi-layer nn
torch.manual_seed(7)
features = torch.randn((1,3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# weights for input to hidden
W1 = torch.randn(n_input, n_hidden)
# weights for hidden to output
W2 = torch.randn(n_hidden, n_output)
# bias terms
b1 = torch.randn((1, n_hidden))
b2 = torch.randn((1, n_output))
# output
h = activation(torch.mm(features, W1) + b1)
output = activation(torch.mm(h, W2) + b2)
print(output)