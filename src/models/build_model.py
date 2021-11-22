from tensorflow import keras

import data_help.data_constants as dc


def build_discriminator():
    model = keras.models.Sequential([
        keras.layers.Dense(512, input_dim=dc.INPUT_LENGTH),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile('adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_generator():
    model = keras.models.Sequential([
        keras.layers.Dense(128, input_dim=dc.LATENT_DIM),
        keras.layers.LeakyReLU(0.3),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(0.3),
        keras.layers.Dense(dc.INPUT_LENGTH, activation='tanh')
    ])

    model.compile('adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_GAN():
    discriminator = build_discriminator()
    generator = build_generator()
    discriminator.trainable = False
    gan = keras.models.Sequential([
        generator,
        discriminator
    ])
    gan.compile('adam', loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return generator, discriminator, gan
