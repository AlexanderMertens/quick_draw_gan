from typing import Tuple
from tensorflow.python.keras.layers.convolutional import Cropping2D, ZeroPadding2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Activation, Conv2DTranspose, Reshape
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizer_v2.adam import Adam

from utility.load_config import load_config


def build_discriminator() -> Sequential:
    config = load_config('config.yaml')

    dimensions = config['dimensions']
    config_d = config['discriminator']

    filters = config_d['filters']
    strides = config_d['strides']
    kernel_size = config_d['kernel_size']
    alpha = config_d['alpha_leaky']
    dropout_rate = config_d['dropout']

    # IN: (None, 28, 28, 1)
    img_input = Input(dimensions['img_shape'])
    x = ZeroPadding2D((2, 2))(img_input)
    # Out: (None, 32, 32, 1)

    x = conv_block(
        x,
        filters,
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True,
        activation=LeakyReLU(alpha)
    )
    # Out: (None, 32, 16, 16, 1)
    x = conv_block(
        x,
        2 * filters,
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True,
        activation=LeakyReLU(alpha)
    )

    # Out: (None, 64, 8, 8, 1)
    x = conv_block(
        x,
        4 * filters,
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True,
        activation=LeakyReLU(alpha)
    )
    # Out: (None, 128, 4, 4, 1)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation=Activation('sigmoid'))(x)

    d_model = Model(img_input, x, name="discriminator")
    lr = config_d['learning_rate']
    d_model.compile(optimizer=Adam(learning_rate=lr, beta_1=0.5), loss='binary_crossentropy',
                    metrics=['binary_accuracy'])
    d_model.summary()
    return d_model


def build_generator() -> Sequential:
    config = load_config('config.yaml')
    config_g = config['generator']
    dimensions = config['dimensions']

    filters = config_g['filters']
    kernel_size = config_g['kernel_size']
    strides = config_g['strides']
    alpha = config_g['alpha_leaky']

    noise = Input(shape=(dimensions['latent'],))
    x = Dense(4 * 4 * filters, use_bias=False)(noise)
    x = Reshape((4, 4, filters))(x)

    # Out: (4, 4, 256)
    x = upsample_block(
        x,
        filters=filters // 2,
        activation=LeakyReLU(alpha),
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True
    )

    # Out: (8, 8, 128)
    x = upsample_block(
        x,
        filters=filters // 4,
        activation=LeakyReLU(alpha),
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True
    )
    # Out: (16, 16, 64)
    x = upsample_block(
        x,
        filters=filters // 8,
        activation=LeakyReLU(alpha),
        kernel_size=kernel_size,
        strides=strides,
        use_bn=True
    )
    # Out: (32, 32, 32)
    x = upsample_block(
        x,
        filters=1,
        kernel_size=kernel_size,
        strides=(1, 1),
        activation=Activation('tanh')
    )
    # Out: (32, 32, 1)
    x = Cropping2D((2, 2))(x)
    # Out: (28, 28, 1)

    g_model = Model(noise, x, name="generator")
    g_model.summary()
    return g_model


def build_GAN() -> Tuple[Sequential, Sequential, Sequential]:
    config_g = load_config('config.yaml')['gan']
    discriminator = build_discriminator()
    generator = build_generator()
    discriminator.trainable = False
    gan = Sequential([
        generator,
        discriminator
    ])
    lr = config_g['learning_rate']
    gan.compile(optimizer=Adam(learning_rate=lr, beta_1=0.5), loss='binary_crossentropy',
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
