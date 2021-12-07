import numpy as np

import data_help.data_constants as dc


def convert_to_image(data: np.array) -> np.array:
    """Reshapes array of data to dimensions of image (28 x 28).

    Args:
        data (np.array): Array of data to be reshaped.

    Returns:
        np.array: Array with dimensions (_, 28, 28, 1)
    """
    return np.reshape(data, (data.shape[0], dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1))


def convert_to_array(data: np.array) -> np.array:
    """Reshapes array of data to flatten the data.

    Args:
        data (np.array): Array of data to be reshaped.

    Returns:
        np.array: Array with dimensions (_, 754)
    """
    return np.reshape(data, (data.shape[0], dc.INPUT_LENGTH))


def flip_images_y(data: np.array) -> np.array:
    """Flip array of images along its y_axis

    Args:
        data (np.array): Array of images to be flipped.

    Returns:
        np.array: Array flipped along its y-axis.
    """
    return np.flip(data, 2)
