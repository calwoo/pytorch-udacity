import numpy as np 
from softmax import softmax

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# is sigmoid = softmax for n=2?
z = [4, 3]
print("sigmoid probability is: ", [sigmoid(z[0]), 1-sigmoid(z[0])])
print("softmax probability is ", softmax(z))