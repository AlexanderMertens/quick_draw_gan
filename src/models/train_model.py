from matplotlib import pyplot as plt
import numpy as np
import data_help.data_constants as dc

from mlflow.tracking.fluent import log_param
from tqdm import tqdm
from data_help.data_transform import convert_to_image
from data_help.make_dataset import generate_random_data, load_data
from models.build_model import build_GAN
from visualization.visualise import plot_images, plot_metrics


def train_model(num_epochs, num_batch=4, batch_size=16):
    log_param('num_epochs', num_epochs)
    log_param('num_batch', num_batch)
    log_param('batch_size', batch_size)

    full_size = num_batch * batch_size

    data = load_data(path=dc.ENVELOPE_DATA_PATH, full_size=full_size)
    real_and_fake = np.concatenate(
        (np.ones((batch_size // 2, 1)), np.zeros((batch_size // 2, 1))))
    ones = np.ones((batch_size, 1))

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
            images_fake = generator.predict(
                generate_random_data(batch_size//2))

            images = np.concatenate((images_real, images_fake))
            d_loss, d_accuracy = discriminator.train_on_batch(
                images, real_and_fake)

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

        if epoch % 1 == 0:
            noise = generate_random_data(100)
            images = convert_to_image(generator.predict(noise))
            plot_images(
                images, path="figures/results/epoch_{}".format(epoch), show=False, save=True)
        # record metrics
        d_loss_avg.append(sum(d_loss_batch) / num_batch)
        d_accuracy_avg.append(sum(d_accuracy_batch) / num_batch)
        g_loss_avg.append(sum(g_loss_batch) / num_batch)
        g_accuracy_avg.append(sum(g_accuracy_batch) / num_batch)

    # plot metrics
    plot_metrics(d_loss_avg=d_loss_avg, d_accuracy_avg=d_accuracy_avg,
                 g_loss_avg=g_loss_avg, g_accuracy_avg=g_accuracy_avg)
    return gan
