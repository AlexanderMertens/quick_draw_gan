from typing import Tuple
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Activation, Conv2DTranspose, Reshape
from tensorflow.python.keras.optimizer_v2.adam import Adam
from azureml.core import Run

import data_help.data_constants as dc


def build_discriminator() -> Sequential:
    run_logger = Run.get_context()

    depth = 32
    dropout = 0.4
    kernel_size = 3
    run_logger.log('Discriminator depth', depth)
    run_logger.log('Discriminator kernel size', kernel_size)
    input_shape = (dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1)

    model = Sequential([
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        # [(W−K+2P)/S]+1
        Conv2D(depth, kernel_size=kernel_size, strides=2,
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(depth*2, kernel_size=kernel_size,
               strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(depth*4, kernel_size=kernel_size,
               strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Out: 1-dim probability
        Flatten(),
        Dense(1),
        Activation('sigmoid'),
    ])

    lr = 0.0001
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_generator() -> Sequential:
    dropout = 0.4
    depth = 256
    dim = 7
    kernel_size = 3
    run_logger = Run.get_context()
    run_logger.log('Generator depth', depth)
    run_logger.log('Generator kernel size', kernel_size)

    model = Sequential([
        Dense(dim*dim*depth, input_dim=dc.LATENT_DIM),
        Dropout(dropout),
        Reshape((dim, dim, depth)),

        Conv2DTranspose(
            int(depth/2), kernel_size=kernel_size, strides=2, padding='same'),
        BatchNormalization(axis=-1, momentum=0.9),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(
            int(depth/4), kernel_size=kernel_size, strides=1, padding='same'),
        BatchNormalization(axis=-1, momentum=0.9),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(
            int(depth/8), kernel_size=kernel_size, strides=1, padding='same'),
        BatchNormalization(axis=-1, momentum=0.9),
        LeakyReLU(alpha=0.2),

        # Out: 28 x 28 x 1 grayscale image [-1.0,1.0] per pix
        Conv2DTranspose(
            1, kernel_size=kernel_size, strides=2, padding='same'),
        Activation('tanh')
    ])

    return model


def build_GAN() -> Tuple[Sequential, Sequential, Sequential]:
    discriminator = build_discriminator()
    generator = build_generator()
    discriminator.trainable = False
    gan = Sequential([
        generator,
        discriminator
    ])
    lr = 0.0002
    gan.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return generator, discriminator, gan
