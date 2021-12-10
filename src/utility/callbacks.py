from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.preprocessing.image import array_to_img

import data_help.data_constants as dc
import tensorflow as tf
import os

from visualization.visualise import plot_images


class GANMonitor(Callback):
    def __init__(self, num_img=100, latent_dim=dc.LATENT_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        os.makedirs("outputs/figures", exist_ok=True)
        plot_images(
            generated_images, path="./outputs/figures/epoch_{:003d}".format(epoch), show=False)
