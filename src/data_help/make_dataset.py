"""Module containing functions that help load and generate data."""
import numpy as np
import data_help.data_constants as dc

from data_help.data_transform import convert_to_image


def load_data(path: str, full_size: int = 1000, verbose: bool = False) -> np.array:
    """Load data at given path.
       Entries of data are resized from 0 to 255 integers to flaots within [-1, 1]....

    Args:
        path (str): Path of location of data.
        full_size (int, optional): Size of the returned dataset. Defaults to 1000.
        verbose (bool, optional): If True, prints shape and type info of data. Defaults to False.

    Returns:
        np.array: array of size full_size with entries in [-1, 1]
    """
    data = simple_load_data(path, end=full_size)
    if verbose:
        print("--data metadata--")
        print(data.shape)
        print(data.dtype)
        print("----------")

    # resize data to fit in [-1, 1]
    data = (data.astype(np.float32) - 127.5) / 127.5
    return convert_to_image(data)


def simple_load_data(path: str, start: int = 0, end: int = None) -> np.array:
    """Load data at given path. Returns all entries between start and end.

    Args:
        path (str): Path of location of data.
        start (int, optional): Index of first entry to be returned. Defaults to 0.
        end (int, optional): Index of last entry to be returned. Defaults to None.

    Returns:
        np.array: Returns numpy array.
    """
    return np.load(path)[start:end]


def generate_random_data(num_samples: int) -> np.array:
    """Generates array of latent vectors randomly generated.

    Args:
        num_samples (int): Size of array to be returned.

    Returns:
        np.array: array of latent vectors.
    """
    return np.random.normal(0, 1, (num_samples, dc.LATENT_DIM))
