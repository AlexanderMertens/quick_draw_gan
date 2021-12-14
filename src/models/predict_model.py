from typing import Tuple
from utility.azure_help import download_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model


def load_models(name: str) -> Tuple[Model, Model]:
    """Loads generators and discriminator corresponding to given name and returns them.

    Args:
        name (str): Name of the registered Azure Model.

    Returns:
        Tuple[Model, Model]: The discriminator and generator.
    """
    download_model('{}-discriminator'.format(name))
    download_model('{}-generator'.format(name))
    print('download finished')
    discriminator = load_model('saved_model/discriminator')
    generator = load_model('saved_model/generator')
    return discriminator, generator
