import numpy as np
import data_help.data_constants as dc

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
