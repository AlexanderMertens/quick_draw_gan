from tensorflow import keras

from data_help.make_dataset import INPUT_LENGTH


def build_discriminator():
    model = keras.models.Sequential([
        keras.models.Dense(512, input_dim=INPUT_LENGTH),
        keras.models.LeakyReLU(0.2),
        keras.models.Dropout(0.3),
        keras.models.Dense(256),
        keras.models.LeakyReLU(0.2),
        keras.models.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile('adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_generator():
    model = keras.models.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(0.3),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(0.3),
        keras.layers.Dense(INPUT_LENGTH, activation='tanh')
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
    gan.compile('adam', loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return generator, discriminator, gan
