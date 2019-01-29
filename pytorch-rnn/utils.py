import numpy as np 


# one hot encoder
def one_hot_encode(int_arr, size_of_vector):
    one_hotized = np.zeros((np.multiply(*int_arr.shape), size_of_vector))
    one_hotized[np.arange(one_hotized.shape[0]), int_arr.flatten()] = 1
    # reshape to original
    one_hotized = one_hotized.reshape((*int_arr.shape, size_of_vector))
    return one_hotized

# batch generator (yay, I'm actually writing a useful generator for once)
def get_batches(data, batch_size, input_size):
    data_len = len(data)
    num_batches = data_len // (batch_size * input_size)
    # to keep things nice and even, just discard trailing chars
    chars_to_keep = num_batches * batch_size * input_size
    data = data[:chars_to_keep]

    # split data into batches
    data = data.reshape((batch_size, -1))
    # yield a batch
    for i in range(0, data.shape[1], input_size):
        x = data[:,i:i+input_size]
        # be careful of the last target iteration!
        y = data[:,i+1:i+input_size+1]
        if i+input_size+1 >= data.shape[1]:
            y = np.c_[data[:,i+1:i+input_size], data[:,0]]
        yield x, y