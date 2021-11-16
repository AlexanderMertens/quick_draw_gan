import numpy as np
from sklearn.model_selection import train_test_split


def load_dogs(full_size=1000, test_size=0.2, path="./data/raw/dog.npy"):
    dogs = np.load(path)[0:full_size]
    print("--dogs metadata--")
    print(dogs.shape)
    print(dogs.dtype)
    print("----------")
    Y = np.full((dogs.shape[0], 2), [1, 0])
    dogs = (dogs.astype(np.float32)) / 255
    dogs = dogs.reshape(dogs.shape[0], 784)
    (x_train, y_train, x_test, y_test) = train_test_split(
        dogs, Y, test_size=test_size)
    return (x_train, y_train, x_test, y_test)
