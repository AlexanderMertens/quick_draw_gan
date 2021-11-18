import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path, full_size=1000, test_size=0.2, verbose=False):
    data = np.load(path)[0:full_size]
    if verbose:
        print("--data metadata--")
        print(data.shape)
        print(data.dtype)
        print("----------")

    # resize data to fit in [0, 1]
    data = (data.astype(np.float32)) / 255
    return data


def split_data(data, full_size=1000, test_size=0.2, verbose=False):
    Y = np.full((data.shape[0], 2), [1, 0])
    (x_train, y_train, x_test, y_test) = train_test_split(
        data, Y, test_size=test_size)
    return (x_train, y_train, x_test, y_test)
