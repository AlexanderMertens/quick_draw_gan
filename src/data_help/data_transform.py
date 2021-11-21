import numpy as np

from data_help.make_dataset import IMAGE_WIDTH, INPUT_LENGTH


def convert_to_image(data):
    return np.reshape(data, (data.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 1))


def convert_to_array(data):
    return np.reshape(data, (data.shape[0], INPUT_LENGTH))
