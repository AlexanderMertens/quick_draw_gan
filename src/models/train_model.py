from matplotlib import pyplot as plt
import numpy as np
import data_help.data_constants as dc

from mlflow.tracking.fluent import log_param
from mlflow.keras import log_model
from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_random_data, load_data
from models.build_model import build_GAN
from visualization.visualise import plot_images, plot_metrics


def train_model(num_epochs, num_batch=4, batch_size=16, run_name=''):
    log_param('num_epochs', num_epochs)
    log_param('num_batch', num_batch)
    log_param('batch_size', batch_size)

    full_size = num_batch * batch_size
    half_batch = batch_size // 2

    data = load_data(path=dc.FILTERED_BUTTERFLY_DATA_PATH, full_size=full_size)
    # real_and_fake = np.concatenate(
    #     (np.ones((batch_size // 2, 1)), np.zeros((batch_size // 2, 1))))
    y_real = np.ones((half_batch, 1))
    y_fake = np.zeros((half_batch, 1))
    ones = np.ones((batch_size, 1))
    fixed_noise = generate_random_data(100)

    generator, discriminator, gan = build_GAN()
    d_loss_avg = []
    d_accuracy_avg = []
    g_loss_avg = []
    g_accuracy_avg = []
    for epoch in range(num_epochs):
        d_loss_batch = []
        d_accuracy_batch = []
        g_loss_batch = []
        g_accuracy_batch = []

        for batch in tqdm(range(num_batch), desc="Epoch {}".format(epoch)):
            discriminator.trainable = True

            # train discriminator
            images_real = data[np.random.randint(
                0, data.shape[0], size=batch_size//2)]
            d_loss_real, d_accuracy_real = discriminator.train_on_batch(
                images_real, y_real)

            images_fake = generator.predict(
                generate_random_data(batch_size//2))
            d_loss_fake, d_accuracy_fake = discriminator.train_on_batch(
                images_fake, y_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_accuracy = (d_accuracy_real + d_accuracy_fake) / 2

            # train generator
            discriminator.trainable = False
            noise = generate_random_data(batch_size)

            g_loss, g_accuracy = gan.train_on_batch(noise, ones)
            print("disc loss:{}, gan loss:{}".format(
                d_loss, g_loss))

            # record metrics
            d_loss_batch.append(d_loss)
            d_accuracy_batch.append(d_accuracy)
            g_loss_batch.append(g_loss)
            g_accuracy_batch.append(g_accuracy)

        images = convert_to_image(generator.predict(fixed_noise))
        plot_images(
            images, path="figures/results/epoch_{:003d}".format(epoch), show=False, save=True)
        if (epoch + 1) % 10 == 0:
            log_model(gan, 'my-model-{}-epoch-{}'.format(run_name, epoch),
                      conda_env='./conda.yaml')
        # record metrics
        d_loss_avg.append(sum(d_loss_batch) / num_batch)
        d_accuracy_avg.append(sum(d_accuracy_batch) / num_batch)
        g_loss_avg.append(sum(g_loss_batch) / num_batch)
        g_accuracy_avg.append(sum(g_accuracy_batch) / num_batch)

    # plot metrics
    plot_metrics(d_loss_avg=d_loss_avg, d_accuracy_avg=d_accuracy_avg,
                 g_loss_avg=g_loss_avg, g_accuracy_avg=g_accuracy_avg)
    return gan
