from matplotlib import pyplot as plt
import numpy as np
import data_help.data_constants as dc

from tensorflow import keras
from mlflow.tracking.fluent import log_param
from mlflow.keras import log_model
from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_fake_data, generate_latent_points, generate_real_data, load_data
from models.build_model import build_GAN
from visualization.visualise import plot_images, plot_metrics


def train_model(num_epochs, n_disc=5, num_batch=4, batch_size=16, run_name=''):
    log_param('n_discriminator', n_disc)
    log_param('num_epochs', num_epochs)
    log_param('num_batch', num_batch)
    log_param('batch_size', batch_size)

    full_size = num_batch * batch_size
    half_batch = batch_size // 2

    data = load_data(path=dc.FILTERED_ENVELOPE_DATA_PATH, full_size=full_size)
    fixed_noise = generate_latent_points(100)

    generator, discriminator, gan = build_GAN()
    d_loss_avg = []
    g_loss_avg = []

    for epoch in range(num_epochs):
        d_loss_batch = []
        g_loss_batch = []

        for batch in tqdm(range(num_batch), desc="Epoch {}".format(epoch)):
            # train discriminator
            discriminator.trainable = True
            d_loss = 0
            for _ in range(n_disc):
                X_real, y_real = generate_real_data(data, half_batch)
                d_loss_real = discriminator.train_on_batch(X_real, y_real)

                X_fake, y_fake = generate_fake_data(generator, half_batch)
                d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
                d_loss += (d_loss_fake + d_loss_real) / 2
            d_loss = d_loss / n_disc

            # train generator
            discriminator.trainable = False
            X_gan = generate_latent_points(batch_size)
            y_gan = -np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(X_gan, y_gan)
            print("disc loss:{}, gan loss:{}".format(
                d_loss, g_loss))

            # record metrics
            d_loss_batch.append(d_loss)
            g_loss_batch.append(g_loss)

        if epoch % 1 == 0:
            images = convert_to_image(generator.predict(fixed_noise))
            plot_images(
                images, path="figures/results/epoch_{}".format(epoch), show=False, save=True)
            log_model(gan, 'my-model-{}-epoch-{}'.format(run_name, epoch),
                      conda_env='./conda.yaml')
        # record metrics
        d_loss_avg.append(sum(d_loss_batch) / num_batch)
        g_loss_avg.append(sum(g_loss_batch) / num_batch)

    # plot metrics
    plot_metrics([d_loss_avg, g_loss_avg], ['Discriminator loss', 'GAN loss'])
    return gan
