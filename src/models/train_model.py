import numpy as np
import data_help.data_constants as dc

from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_random_data, load_data, split_data
from models.build_model import build_GAN
from visualization.visualise import plot_images


def train_model(num_epochs, batch_size=256):
    num_batch = 16
    n = (num_batch * 5) // 8
    full_size = n * batch_size

    data = load_data(path=dc.DOG_DATA_PATH, full_size=full_size)
    y = np.ones(full_size)
    fake_data = generate_random_data(full_size)
    fake_y = np.zeros(full_size)

    X_train, X_val, y_train, y_val = split_data(
        np.concatenate((data, fake_data)), np.concatenate((y, fake_y)), test_size=1/5)

    generator, discriminator, gan = build_GAN()
    for epoch in range(num_epochs):
        for batch in tqdm(range(num_batch), desc="Epoch {}".format(epoch)):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X = X_train[start:end]
            y = y_train[start:end]

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y)

            noise = generate_random_data(batch_size)
            y = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y)
            print("disc loss:{}, gan loss:{}".format(d_loss, g_loss))

    noise = generate_random_data(100)
    prediction = generator.predict(noise)
    images = convert_to_image(prediction)
    plot_images(images)
