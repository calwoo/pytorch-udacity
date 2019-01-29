import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

# create data
plt.figure(figsize=(8,5))
seq_length = 20
timesteps = np.linspace(0, np.pi, seq_length+1)
data = np.sin(timesteps)
data = data[:,np.newaxis]

x = data[:-1]
y = data[1:]

plt.plot(timesteps[1:], x, 'r.', label='input, x') # x
plt.plot(timesteps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
plt.show()

"""
We will create an RNN model in PyTorch. We won't build it from scratch, instead we'll use the RNN cell from the
nn module, which is pretty easy to use. Same with the LSTM. I will implement them from scratch (ie, numpy) in
another github repo.
"""
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

# hyperparameters
input_size=1 
output_size=1
hidden_dim=32
n_layers=1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

def train(rnn, epochs):
    # initial hidden state
    hidden = None
    for e in tqdm(range(epochs)):
        timesteps = np.linspace(e*np.pi, (e+1)*np.pi, seq_length+1)
        data = np.sin(timesteps)
        data.resize((seq_length+1, 1))
        x = data[:-1]
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y = data[1:]
        y_tensor = torch.Tensor(y)

        # forward
        out, hidden = rnn.forward(x_tensor, hidden)
        # detach the hidden state from its history so we don't backprop through it all
        hidden = hidden.data
        # loss
        loss = criterion(out, y_tensor)
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        if e % 10 == 0:
            print("epoch %d -- loss: %f" % (e, loss.item()))
            plt.plot(timesteps[1:], out.data.numpy().flatten(), 'b.') # predictions
            plt.show()

    return rnn

# train the rnn and monitor results
epochs = 100
trained_rnn = train(rnn, epochs)
