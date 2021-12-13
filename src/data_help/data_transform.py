import numpy as np

from utility.load_config import load_config


def convert_to_image(data: np.array) -> np.array:
    """Reshapes array of data to dimensions of image as specified in config.yaml.

    Args:
        data (np.array): Array of data to be reshaped.

    Returns:
        np.array: Array with dimensions specified in config.yaml.
    """
    dimensions = load_config('config.yaml')['dimensions']
    return np.reshape(data, (data.shape[0], *dimensions['img_shape']))


def convert_to_array(data: np.array) -> np.array:
    """Reshapes array of data to flatten the data.

    Args:
        data (np.array): Array of data to be reshaped.

    Returns:
        np.array: Array with dimensions as specified in config.yaml.
    """
    dimensions = load_config('config.yaml')['dimensions']
    return np.reshape(data, (data.shape[0], dimensions['input_length']))


def flip_images_y(data: np.array) -> np.array:
    """Flip array of images along its y_axis

    Args:
        data (np.array): Array of images to be flipped.

    Returns:
        np.array: Array flipped along its y-axis.
    """
    return np.flip(data, 2)
