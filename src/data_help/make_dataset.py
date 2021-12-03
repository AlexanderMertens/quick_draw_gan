import numpy as np
import data_help.data_constants as dc
from sklearn.model_selection import train_test_split

from data_help.data_transform import convert_to_image


def load_data(path, full_size=1000, verbose=False):
    data = simple_load_data(path, end=full_size)
    if verbose:
        print("--data metadata--")
        print(data.shape)
        print(data.dtype)
        print("----------")

    # resize data to fit in [-1, 1]
    data = (data.astype(np.float32) - 127.5) / 127.5
    return convert_to_image(data)


def simple_load_data(path, start=0, end=None):
    return np.load(path)[start:end]


def generate_random_data(num_samples):
    return np.random.normal(0, 1, (num_samples, dc.LATENT_DIM))


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
