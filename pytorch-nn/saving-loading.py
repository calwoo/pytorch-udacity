"""
In PyTorch, we generally don't always want to keep training our model from scratch every time. Usually we'd
really like to save our parameters and load them for use later.

The parameters for PyTorch networks are saved in a model's state_dict.
"""

# to save a model, we use
torch.save(model.state_dict, "<checkpoint path>")

# to load it again, we set
state_dict = torch.load("<checkpoint path>")
model.load_state_dict(state_dict)

# information about the model architecture must be saved in the checkpoint
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, "<checkpoint path>")


