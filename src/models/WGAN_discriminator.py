from typing import Tuple
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model


import data_help.data_constants as dc


def get_discriminator_model():
    img_input = Input(shape=dc.IMG_SHAPE)
    filters = 32
    # Zero pad the input to make the input images size to (32, 32, 1).
    x = ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        filters,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
    )
    # Output: (None, 64, 16, 16)
    x = conv_block(
        x,
        2 * filters,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_dropout=True,
        drop_value=0.3,
    )
    # Output: (None, 128, 8, 8)
    x = conv_block(
        x,
        4 * filters,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_dropout=True,
        drop_value=0.3,
    )
    # Output: (None, 256, 4, 4)
    x = conv_block(
        x,
        8 * filters,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_dropout=False,
        drop_value=0.3,
    )
    # Output: (None, 256, 2, 2)

    x = Flatten()(x)
    # Output: (None, 2 * 2 * 256)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)

    d_model = Model(img_input, x, name="discriminator")
    return d_model


def conv_block(
    x: Layer,
    filters: int,
    activation: Layer = LeakyReLU(0.2),
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
