import numpy as np
import data_help.data_constants as dc

from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_random_data, load_data, split_data
from models.build_model import build_GAN
from visualization.visualise import plot_images


def train_model(num_epochs, num_batch=4, batch_size=16):
    full_size = num_batch * batch_size

    data = load_data(path=dc.DOG_DATA_PATH, full_size=full_size)

    generator, discriminator, gan = build_GAN()
    for epoch in range(num_epochs):
        for batch in tqdm(range(num_batch), desc="Epoch {}".format(epoch)):
            discriminator.trainable = True

            # train discriminator
            images_real = data[np.random.randint(
                0, data.shape[0], size=batch_size)]
            images_fake = generator.predict(generate_random_data(batch_size))

            X = np.concatenate((images_real, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = discriminator.train_on_batch(X, y)[1]

            # train generator
            discriminator.trainable = False
            noise = generate_random_data(batch_size)
            y = np.ones(batch_size)
            g_loss = gan.train_on_batch(noise, y)[1]
            print("disc loss:{}, gan loss:{}".format(
                d_loss, g_loss))

    noise = generate_random_data(100)
    prediction = generator.predict(noise)
    images = convert_to_image(prediction)
    plot_images(images)
