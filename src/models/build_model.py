from tensorflow import keras
from mlflow import log_param

import data_help.data_constants as dc


def build_discriminator():
    depth = 32
    dropout = 0.4
    kernel_size = 3
    input_shape = (dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1)

    log_param('Discriminator depth', depth)
    log_param('Discriminator dropout rate', dropout)
    log_param('Discriminator kernel size', kernel_size)
    model = keras.models.Sequential([
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        # [(W−K+2P)/S]+1
        keras.layers.Conv2D(depth, kernel_size=kernel_size, strides=2,
                            input_shape=input_shape, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.02),

        keras.layers.Conv2D(depth*2, kernel_size=kernel_size,
                            strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.02),

        keras.layers.Conv2D(depth*4, kernel_size=kernel_size,
                            strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.02),

        # Out: 1-dim probability
        keras.layers.Flatten(),
        keras.layers.Dense(1),
        keras.layers.Activation('sigmoid'),
    ])

    lr = 0.0001
    log_param('Discriminator learning rate', lr)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_generator():
    dropout = 0.4
    depth = 256
    dim = 7
    kernel_size = 3

    log_param('Generator depth', depth)
    log_param('Generator dropout rate', dropout)

    model = keras.models.Sequential([
        keras.layers.Dense(dim*dim*depth, input_dim=dc.LATENT_DIM),
        keras.layers.Reshape((dim, dim, depth)),

        keras.layers.Conv2DTranspose(
            int(depth/2), kernel_size=kernel_size, strides=2, padding='same'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.LeakyReLU(alpha=0.02),

        keras.layers.Conv2DTranspose(
            int(depth/4), kernel_size=kernel_size, strides=1, padding='same'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.LeakyReLU(alpha=0.02),

        # Out: 28 x 28 x 1 grayscale image [-1.0,1.0] per pix
        keras.layers.Conv2DTranspose(
            1, kernel_size=kernel_size, strides=2, padding='same'),
        keras.layers.Activation('tanh')
    ])

    return model


def build_GAN():
    discriminator = build_discriminator()
    generator = build_generator()
    discriminator.trainable = False
    gan = keras.models.Sequential([
        generator,
        discriminator
    ])
    lr = 0.0002
    log_param('Generator learning rate', lr)
    gan.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return generator, discriminator, gan
