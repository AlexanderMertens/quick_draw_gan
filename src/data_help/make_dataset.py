import numpy as np
import data_help.data_constants as dc
from sklearn.model_selection import train_test_split

from data_help.data_transform import convert_to_array


def load_data(path, full_size=1000, verbose=False):
    data = np.load(path)[0:full_size]
    if verbose:
        print("--data metadata--")
        print(data.shape)
        print(data.dtype)
        print("----------")

    # resize data to fit in [0, 1]
    data = (data.astype(np.float32)) / 255
    return convert_to_array(data)


def generate_random_data(num_samples):
    return np.random.normal(0.5, 0.5, (num_samples, dc.INPUT_LENGTH))


def split_data(data, Y, test_size=0.2):
    return train_test_split(data, Y, test_size=test_size)


def create_cats_and_dogs_data(size=5000):
    cats_data = load_data(path=dc.CAT_DATA_PATH, full_size=size)
    cats_y = np.zeros(size)
    dogs_data = load_data(path=dc.DOG_DATA_PATH, full_size=size)
    dogs_y = np.ones(size)
    data = np.concatenate((cats_data, dogs_data))
    Y = np.concatenate((cats_y, dogs_y))
    return split_data(data, Y)
