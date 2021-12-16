from typing import Tuple
from PIL import Image as im
from PIL import ImageChops

import numpy as np
from data_help.make_dataset import generate_random_data
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


def generate_images(generator: Model, num_samples=10) -> np.array:
    """Generates an array of images generated using the provided generator model.

    Args:
        generator (Model): The generator used to generate images.
        num_samples (int, optional): Amount of images to generate. Defaults to 10.

    Returns:
        np.array: Data of the generated images.
    """
    noise = generate_random_data(num_samples=num_samples)
    # Outputs images with pixels in [-1, 1]
    images = np.array(generator(noise))
    return images


def image_float_to_int(images: np.array) -> np.array:
    """Converts image data that consists of floats to integers. 
    Reshapes image data in appropriate shape for saving.

    Args:
        images (np.array): The images to be converted.

    Returns:
        np.array: Converted images.
    """
    # Convert pixels to integers 0 to 255
    images = (images * 127.5 + 127.5).astype(np.uint8)
    # Reshape to array of 2d arrays
    images = images.reshape((images.shape[0:3]))
    return images


def filter_images(images: np.array, discriminator: Model, quality: float) -> np.array:
    # Probabilities the images are real according to discriminator.
    # i.e. quality of the image.
    probs = np.array(discriminator(images))
    # Filter any images of too low a quality
    filter = [probability[0] > quality for probability in probs]
    return images[filter]


def generate_quality_images(generator: Model, discriminator: Model, num_samples: int, quality: float = 0.5) -> np.array:
    """Generate only images of appropriate quality.

    Args:
        generator (Model): Generator model that generators images.
        discriminator (Model): Discriminator model that judges images quality.
        num_samples (int): Number of samples to be generated.
        quality (float): Float between 0.0 and 1.0 representing quality of the images. Defaults to 0.5

    Returns:
        np.array: Array containing images.
    """
    quality_images = filter_images(
        generate_images(generator, 100), discriminator, quality=quality)
    while quality_images.shape[0] < num_samples:
        new_images = filter_images(
            generate_images(generator, 100), discriminator, quality=quality)
        quality_images = np.concatenate((quality_images, new_images))
        print(quality_images.shape)
    return quality_images[:num_samples]


def save_images(images: np.array):
    """Save images to outputs/figures folder using PILLOW.

    Args:
        images (np.array): Data of images to be saved.
    """
    for image_data, n in zip(images, range(len(images))):
        image = im.fromarray(image_data)
        image = ImageChops.invert(image)
        image.save('./outputs/figures/img_{:003d}.png'.format(n), format='png')
