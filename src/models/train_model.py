import numpy as np
import tensorflow as tf
import sys

from azureml.core import Run
from data_help.data_transform import convert_to_image, flip_images_y
from data_help.make_dataset import generate_random_data, load_data
from models.build_model import build_GAN
from visualization.visualise import plot_images, plot_metrics
from tqdm import tqdm

DISC_LOSS_ID = 0
DISC_ACC_ID = 1
GEN_LOSS_ID = 2
GEN_ACC_ID = 3


def train_model(data_path: str, num_epochs: int, num_batch: int = 4, batch_size: int = 16):
    run_logger = Run.get_context()

    full_size = num_batch * batch_size
    half_batch = batch_size // 2

    data = load_data(path=data_path, full_size=full_size // 2)
    # flip images along y axis and shuffle them among original data
    data = np.concatenate((data, flip_images_y(data)), axis=0)
    np.random.shuffle(data)

    y_real = np.ones((half_batch, 1))
    y_fake = np.zeros((half_batch, 1))
    ones = np.ones((batch_size, 1))

    # Create fixed noise to generate images
    fixed_noise = generate_random_data(100)

    generator, discriminator, gan = build_GAN()
    metrics_avg = [[] for _ in range(4)]
    for epoch in range(num_epochs):
        print('\n-------------------- epoch {} --------------------'.format(epoch))
        metrics_batch = [[] for _ in range(4)]

        # batch loop
        for batch in tqdm(range(num_batch),
                          file=sys.stdout, desc='epoch {}'.format(epoch)):
            metrics = np.zeros(4)

            # train discriminator
            discriminator.trainable = True
            images_real = data[np.random.randint(
                0, data.shape[0], size=half_batch)]
            metrics_real = discriminator.train_on_batch(
                images_real, y_real)

            images_fake = generator.predict(
                generate_random_data(half_batch))
            metrics_fake = discriminator.train_on_batch(images_fake, y_fake)
            # take average of real and fake loss
            metrics[DISC_LOSS_ID:DISC_ACC_ID + 1] = [(d_real + d_fake) / 2 for d_real,
                                                     d_fake in zip(metrics_real, metrics_fake)]

            # train generator
            discriminator.trainable = False
            noise = generate_random_data(batch_size)

            metrics[GEN_LOSS_ID:GEN_ACC_ID + 1] = gan.train_on_batch(
                noise, ones)

            print("\nbatch: {} / {} -- disc loss:{}, gan loss:{}".format(batch +
                  1, num_batch, metrics[DISC_LOSS_ID], metrics[GEN_LOSS_ID]))

            # record metrics
            for metric_array, metric_value in zip(metrics_batch, metrics):
                metric_array.append(metric_value)

        images = convert_to_image(generator.predict(fixed_noise))
        plot_images(
            images, path="./outputs/figures/epoch_{:003d}".format(epoch), show=False)
        # Save models
        if (epoch + 1) % 10 == 0:
            gen_path = './outputs/generator_epoch_{:003d}'.format(epoch + 1)
            disc_path = './outputs/discriminator_epoch_{:003d}'.format(
                epoch + 1)
            tf.saved_model.save(generator, gen_path)
            tf.saved_model.save(discriminator, disc_path)

        # record metrics
        for metric_array, batch_array in zip(metrics_avg, metrics_batch):
            metric_array.append(sum(batch_array) / num_batch)

    # plot metrics
    plot_metrics(d_loss_avg=metrics_avg[DISC_LOSS_ID], d_accuracy_avg=metrics_avg[DISC_ACC_ID],
                 g_loss_avg=metrics_avg[GEN_LOSS_ID], g_accuracy_avg=metrics_avg[GEN_ACC_ID])
    run_logger.log_list('Discriminator loss', metrics_avg[DISC_LOSS_ID])
    run_logger.log_list('Discriminator accuracy', metrics_avg[DISC_ACC_ID])
    run_logger.log_list('GAN loss', metrics_avg[GEN_LOSS_ID])
    run_logger.log_list('GAN accuracy', metrics_avg[GEN_ACC_ID])
    return discriminator, generator
