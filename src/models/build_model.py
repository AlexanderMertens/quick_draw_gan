from tensorflow import keras
from mlflow import log_param

import data_help.data_constants as dc


def build_discriminator():
    depth = 64
    dropout = 0.4
    kernel_size = 5
    constraint = ClipConstraint(0.01)
    input_shape = (dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, 1)

    log_param('Discriminator depth', depth)
    log_param('Discriminator dropout rate', dropout)
    log_param('Discriminator kernel size', kernel_size)
    model = keras.models.Sequential([
        # In: (28, 28, 1)
        keras.layers.Conv2D(depth, kernel_size=kernel_size, strides=2, kernel_constraint=constraint,
                            input_shape=input_shape, padding='same'),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(depth*2, kernel_size=kernel_size, kernel_constraint=constraint,
                            strides=2, padding='same'),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(depth*4, kernel_size=kernel_size, kernel_constraint=constraint,
                            strides=2, padding='same'),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(depth*8, kernel_size=kernel_size, kernel_constraint=constraint,
                            strides=1, padding='same'),
        keras.layers.Dropout(dropout),

        keras.layers.Flatten(),
        keras.layers.Dense(1),
    ])

    lr = 0.00005
    log_param('Discriminator learning rate', lr)
    opt = keras.optimizers.RMSprop(lr=lr)
    model.compile(optimizer=opt, loss=wasserstein_loss)
    return model


def build_generator():
    dropout = 0.4
    depth = 256
    dim = 7

    log_param('Generator depth', depth)
    log_param('Generator dropout rate', dropout)

    model = keras.models.Sequential([
        keras.layers.Dense(dim*dim*depth, input_dim=dc.LATENT_DIM),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Reshape((dim, dim, depth)),
        keras.layers.Dropout(dropout),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(int(depth/2), 5, padding='same'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.Activation('relu'),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(int(depth/4), 5, padding='same'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.Activation('relu'),

        keras.layers.Conv2DTranspose(int(depth/8), 5, padding='same'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9),
        keras.layers.Activation('relu'),

        # Out: 28 x 28 x 1 grayscale image [-1, 1] per pix
        keras.layers.Conv2DTranspose(1, 5, padding='same'),
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
    lr = 0.00005
    log_param('Generator learning rate', lr)
    opt = keras.optimizers.RMSprop(lr=lr)
    gan.compile(optimizer=opt, loss=wasserstein_loss)
    return generator, discriminator, gan


def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)


class ClipConstraint(keras.constraints.Constraint):
    # clip model weights to a given hypercube
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
