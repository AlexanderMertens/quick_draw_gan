import numpy as np
from numpy.core.fromnumeric import size
from sklearn.model_selection import train_test_split

DOG_DATA_PATH = "./data/raw/dog.npy"
CAT_DATA_PATH = "./data/raw/cat.npy"


def load_data(path, full_size=1000, verbose=False):
    data = np.load(path)[0:full_size]
    if verbose:
        print("--data metadata--")
        print(data.shape)
        print(data.dtype)
        print("----------")

    # resize data to fit in [0, 1]
    data = (data.astype(np.float32)) / 255
    return np.reshape(data, (full_size, 28, 28))


def split_data(data, Y, test_size=0.2):
    (x_train, y_train, x_test, y_test) = train_test_split(
        data, Y, test_size=test_size)
    return (x_train, y_train, x_test, y_test)
