import numpy as np
from sklearn.model_selection import train_test_split


def load_dogs(full_size=1000, test_size=0.2, path="./data/raw/dog.npy", verbose=False):
    dogs = np.load(path)[0:full_size]
    if verbose:
        print("--dogs metadata--")
        print(dogs.shape)
        print(dogs.dtype)
        print("----------")

    dogs = (dogs.astype(np.float32)) / 255
    dogs = dogs.reshape(dogs.shape[0], 784)
    return dogs


def split_data(full_size=1000, test_size=0.2, path="./data/raw/dog.npy", verbose=False):
    dogs = load_dogs(full_size=full_size, test_size=test_size,
                     path=path, verbose=verbose)
    Y = np.full((dogs.shape[0], 2), [1, 0])
    (x_train, y_train, x_test, y_test) = train_test_split(
        dogs, Y, test_size=test_size)
    return (x_train, y_train, x_test, y_test)
