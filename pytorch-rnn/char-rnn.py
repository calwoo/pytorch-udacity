import torch
import numpy as np 
from torch import nn
import torch.nn.functional as F 

from utils import *
from tqdm import tqdm

# load data
with open("anna.txt", "r") as txt:
    text = txt.read()

# following karpathy's char-rnn gist, we tokenize the data (one-hot-ize?)
# we don't know how many unique chars appear in the data, so just count them straight from text
unique_chars = tuple(set(text))
num_unique_chars = len(unique_chars)
int2char = dict(enumerate(unique_chars))
char2int = {ch:i for i, ch in int2char.items()}

# encode text
encoded_text = np.array([char2int[ch] for ch in text])

# or char-RNN model
class CharRNN(nn.Module):
    def __init__(self, tokens, hidden_dim=256, n_layers=2, dropout_prob=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        # char dictionaries
        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.int2token = dict(enumerate(self.tokens))
        self.token2int = {tk:i for i, tk in self.int2token.items()}

        # RNN model itself
        self.lstm = nn.LSTM(self.num_tokens, self.hidden_dim, self.n_layers,
            dropout=self.dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(self.hidden_dim, self.num_tokens)

    def forward(self, x, hidden):
        # each input will pass through LSTM cell and get a softmax
        # output to get a probability distribution over tokens
        out, hidden = self.lstm(x.float(), hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def initialize_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# training loop
def train(model, data, epochs=10, batch_size=10, seq_length=30, lr=0.001, clip=5, val_frac=0.1):
    # set to training mode, ie, turn on dropout layers
    model.train()
    # optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # create training and validation data
    val_idx = int(len(data) * (1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    # loooooop
    for e in tqdm(range(epochs)):
        # initialize hidden state
        counter = 0
        hidden = model.initialize_hidden_state(batch_size)
        for x, y in tqdm(get_batches(data, batch_size, seq_length)):
            counter += 1
            # preprocess data
            x = one_hot_encode(x, model.num_tokens)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            # to prevent the hidden state backproping through all time, we
            # just store current hidden state in new vars
            hidden = tuple([each.data for each in hidden])
            # zero grads
            optimizer.zero_grad()
            # forward
            out, hidden = model.forward(x, hidden)
            loss = criterion(out, y.contiguous().view(batch_size * seq_length).long())
            # backward
            loss.backward()
            # clip gradients so they don't vanish
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # print loss
            if counter % 10 == 0:
                # Get validation loss
                val_h = model.initialize_hidden_state(batch_size)
                val_losses = []
                model.eval() # set model into evaluation mode-- kill off dropout
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, model.num_tokens)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    out, val_h = model.forward(x, val_h)
                    val_loss = criterion(out, y.contiguous().view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                model.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                        "Loss: {:.4f}...".format(loss.item()),
                        "Val Loss: {:.4f}".format(np.mean(val_losses)))
                # save point
                torch.save(model.state_dict(), "ckpt")

# test it!
model = CharRNN(unique_chars)
train(model, encoded_text, batch_size=128, seq_length=100)
