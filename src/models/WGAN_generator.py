from typing import Tuple
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Cropping2D
from tensorflow.python.keras.models import Model


import data_help.data_constants as dc


def get_generator_model():
    filters = 256
    noise = Input(shape=(dc.LATENT_DIM,))
    x = Dense(4 * 4 * filters, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Reshape((4, 4, filters))(x)
    # Output: (None, 256, 4, 4)
    x = upsample_block(
        x,
        filters // 2,
        use_bn=True,
    )
    # Output: (None, 128, 8, 8)
    x = upsample_block(
        x,
        filters // 2,
        use_bn=True,
    )
    # Output: (None, 128, 16, 16)
    x = upsample_block(
        x, 1, Activation("tanh"), use_bn=True
    )
    # Output: (None, 1, 32, 32)

    # Crop image to (28, 28)
    x = Cropping2D((2, 2))(x)

    g_model = Model(inputs=noise, outputs=x, name="generator")
    return g_model


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
