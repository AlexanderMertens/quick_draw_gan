import numpy as np

import data_help.data_constants as dc


def convert_to_image(data):
    return np.reshape(data, (data.shape[0], dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1))


def convert_to_array(data):
    return np.reshape(data, (data.shape[0], dc.INPUT_LENGTH))
