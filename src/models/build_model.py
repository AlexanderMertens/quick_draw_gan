from tensorflow import keras


def build_cats_and_dogs_discriminator(num_filters=8, filter_size=3, pool_size=2):
    model = keras.models.Sequential([
        keras.layers.Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model
