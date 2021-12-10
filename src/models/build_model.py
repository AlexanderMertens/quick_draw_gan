import tensorflow as tf
import data_help.data_constants as dc

from tensorflow.python.keras.optimizer_v2.adam import Adam
from models.WGAN import WGAN
from models.WGAN_discriminator import get_discriminator_model
from models.WGAN_generator import get_generator_model


def build_wgan():
    generator_optimizer = Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    wgan = WGAN(
        discriminator=get_discriminator_model(),
        generator=get_generator_model(),
        latent_dim=dc.LATENT_DIM
    )

    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )
    return wgan


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)
