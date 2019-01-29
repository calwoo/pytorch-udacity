import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt 

# create data
plt.figure(figsize=(8,5))
seq_length = 20
timesteps = np.linspace(0, np.pi, seq_length+1)
data = np.sin(timesteps)
data = data[:,np.newaxis]

x = data[:-1]
y = data[1:]

plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
plt.show()

"""
We will create an RNN model in PyTorch. We won't build it from scratch, instead we'll use the RNN cell from the
nn module, which is pretty easy to use. Same with the LSTM. I will implement them from scratch (ie, numpy) in
another github repo.
"""
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        