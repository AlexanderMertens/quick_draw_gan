from tensorflow import keras


def build_discriminator(num_filters=8, filter_size=3, pool_size=2):
    model = keras.models.Sequential([
        keras.layers.Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        keras.layers.Flatten(),
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
        keras.layers.Dense(784, activation='tanh')
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
