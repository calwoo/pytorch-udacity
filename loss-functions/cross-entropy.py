import numpy as np 

def cross_entropy(Y, probs):
    cross_entropy = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            cross_entropy += -np.log(probs[i])
        else:
            cross_entropy += -np.log(1-probs[i])
    return cross_entropy