import numpy as np 

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps)


# Test
logits = [5,6,7]
print(softmax(logits))