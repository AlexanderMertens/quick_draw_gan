import numpy as np
import data_help.data_constants as dc

from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_random_data, load_data
from models.build_model import build_GAN
from visualization.visualise import plot_images


def train_model(num_epochs, num_batch=4, batch_size=16):
    full_size = num_batch * batch_size

    data = load_data(path=dc.ENVELOPE_DATA_PATH, full_size=full_size)
    real = np.ones((batch_size // 2, 1))
    fake = np.zeros((batch_size // 2, 1))

    ones = np.ones((batch_size, 1))

    generator, discriminator, gan = build_GAN()
    for epoch in range(num_epochs):
        for batch in tqdm(range(num_batch), desc="Epoch {}".format(epoch)):
            discriminator.trainable = True

            # train discriminator
            images_real = data[np.random.randint(
                0, data.shape[0], size=batch_size//2)]
            d_loss_real = discriminator.train_on_batch(images_real, real)

            images_fake = generator.predict(
                generate_random_data(batch_size//2))
            d_loss_fake = discriminator.train_on_batch(
                images_fake, fake)

            # train generator
            discriminator.trainable = False
            noise = generate_random_data(batch_size)

            g_loss = gan.train_on_batch(noise, ones)[1]
            print("disc loss real:{}, disc loss fake:{}, gan loss:{}".format(
                d_loss_real, d_loss_fake, g_loss))

        if epoch % 1 == 0:
            noise = generate_random_data(100)
            images = convert_to_image(generator.predict(noise))
            plot_images(
                images, path="figures/results/epoch_{}".format(epoch), show=False, save=True)
