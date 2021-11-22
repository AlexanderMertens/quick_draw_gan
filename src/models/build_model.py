from tensorflow import keras

import data_help.data_constants as dc


def build_discriminator():
    depth = 64
    dropout = 0.4
    input_shape = (dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1)

    model = keras.models.Sequential([
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        # [(W−K+2P)/S]+1
        keras.layers.Conv2D(depth, kernel_size=4, strides=2,
                            input_shape=input_shape, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(depth*2, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(depth*2, kernel_size=4, strides=1, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(dropout),

        # Out: 1-dim probability
        keras.layers.Flatten(),
        keras.layers.Dense(1),
        keras.layers.Activation('sigmoid'),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def build_generator():
    dropout = 0.4
    depth = 256
    dim = 7

    model = keras.models.Sequential([
        keras.layers.Dense(dim*dim*depth, input_dim=dc.LATENT_DIM),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Reshape((dim, dim, depth)),
        keras.layers.Dropout(dropout),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(int(depth/2), 5, padding='same'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(int(depth/4), 5, padding='same'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),

        keras.layers.Conv2DTranspose(int(depth/8), 5, padding='same'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        keras.layers.Conv2DTranspose(1, 5, padding='same'),
        keras.layers.Activation('sigmoid')
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
    gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return generator, discriminator, gan
