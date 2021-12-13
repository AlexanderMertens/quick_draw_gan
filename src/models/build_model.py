from typing import Tuple
from tensorflow.python.keras.layers.convolutional import Cropping2D, ZeroPadding2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Activation, Conv2DTranspose, Reshape
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizer_v2.adam import Adam
from azureml.core import Run

import data_help.data_constants as dc


def build_discriminator() -> Sequential:
    run_logger = Run.get_context()

    filters = 32
    strides_2 = (2, 2)
    kernel_size_3 = (3, 3)
    run_logger.log('Discriminator filters', filters)
    run_logger.log('Discriminator kernel size', kernel_size_3)

    # IN: (None, 28, 28, 1)
    img_input = Input(dc.IMG_SHAPE)
    x = ZeroPadding2D((2, 2))(img_input)
    # Out: (None, 32, 32, 1)

    x = conv_block(
        x,
        filters,
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True,
        activation=LeakyReLU(0.2)
    )
    # Out: (None, 32, 16, 16, 1)
    x = conv_block(
        x,
        2 * filters,
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True,
        activation=LeakyReLU(0.2)
    )

    # Out: (None, 64, 8, 8, 1)
    x = conv_block(
        x,
        4 * filters,
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True,
        activation=LeakyReLU(0.2)
    )
    # Out: (None, 128, 4, 4, 1)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    d_model = Model(img_input, x, name="discriminator")
    lr = 0.0001
    d_model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                    metrics=['binary_accuracy'])
    d_model.summary()
    return d_model


def build_generator() -> Sequential:
    filters = 256
    kernel_size_3 = (3, 3)
    strides_2 = (2, 2)
    run_logger = Run.get_context()
    run_logger.log('Generator filters', filters)
    run_logger.log('Generator kernel size', kernel_size_3)

    noise = Input(shape=(dc.LATENT_DIM,))
    x = Dense(4 * 4 * filters, use_bias=False)(noise)
    x = Reshape((dc.IMAGE_WIDTH, dc.IMAGE_WIDTH, filters))(x)

    # Out: (4, 4, 256)
    x = upsample_block(
        x,
        filters=filters // 2,
        activation=LeakyReLU(0.2),
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True
    )

    # Out: (8, 8, 128)
    x = upsample_block(
        x,
        filters=filters // 4,
        activation=LeakyReLU(0.2),
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True
    )
    # Out: (16, 16, 64)
    x = upsample_block(
        x,
        filters=filters // 8,
        activation=LeakyReLU(0.2),
        kernel_size=kernel_size_3,
        strides=strides_2,
        use_bn=True
    )
    # Out: (32, 32, 32)
    x = upsample_block(
        x,
        filters=1,
        kernel_size=kernel_size_3,
        strides=(1, 1),
        activation='tanh'
    )
    # Out: (32, 32, 1)
    x = Cropping2D((2, 2))(x)
    # Out: (28, 28, 1)

    g_model = Model(noise, x, name="generator")
    g_model.summary()
    return g_model


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


def conv_block(
    x: Layer,
    filters: int,
    activation: Layer,
    kernel_size: Tuple[int, int] = (3, 3),
    strides: Tuple[int, int] = (1, 1),
    padding: str = "same",
    use_bias: bool = True,
    use_bn: bool = False,
    use_dropout: bool = False,
    drop_value: float = 0.5,
) -> Layer:
    """Block of layers containing convolutional layer to insert into discriminator.

    Args:
        x (Layer): The previous layer.
        filters (int): Dimensionality of output space.
        activation (Layer, optional): The activation layer. Defaults to LeakyReLU(0.2).
        kernel_size (Tuple[int, int], optional): Specifies shape convolutional window. Defaults to (3, 3).
        strides (Tuple[int, int], optional): Specifies the strides. Defaults to (1, 1).
        padding (str, optional): "valid" or "same". Defaults to "same".
        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to True.
        use_bn (bool, optional): Whether the block uses batch normalization. Defaults to False.
        use_dropout (bool, optional): Whether the block uses a dropout layer. Defaults to False.
        drop_value (float, optional): Dropout value. Defaults to 0.5.

    Returns:
        Layer: Returns convolutional block.
    """
    x = Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def upsample_block(
    x: Layer,
    filters: int,
    activation: Layer = LeakyReLU(0.2),
    kernel_size: Tuple[int, int] = (3, 3),
    strides: Tuple[int, int] = (2, 2),
    padding: str = "same",
    use_bn: bool = False,
    use_bias: bool = True,
    use_dropout: bool = False,
    drop_value: float = 0.3,
) -> Layer:
    """Upsampling using Conv2dTrans layer with strides.

    Args:
        x (Layer): The previous layer.
        filters (int): Dimensionality of output space.
        activation (Layer, optional): The activation layer. Defaults to LeakyReLU(0.2).
        kernel_size (Tuple[int, int], optional): Specifies shape convolutional window. Defaults to (3, 3).
        strides (Tuple[int, int], optional): Specifies the strides. Defaults to (2, 2).
        padding (str, optional): "valid" or "same". Defaults to "same".
        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to True.
        use_bn (bool, optional): Whether the block uses batch normalization. Defaults to False.
        use_dropout (bool, optional): Whether the block uses a dropout layer. Defaults to False.
        drop_value (float, optional): Dropout value. Defaults to 0.5.

    Returns:
        Layer: Returns upsampling block.
    """
    x = Conv2DTranspose(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x
