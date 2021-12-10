import argparse
import logging
import os
import tensorflow as tf
from data_help.make_dataset import load_data
from models.build_model import build_wgan

from utility.callbacks import GANMonitor

if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    os.makedirs('./outputs/figures')

    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument('-p', '--path', type=str,
                        required=True, help="Path to location of data")
    parser.add_argument('-e', '--epochs', type=int,
                        default=10, help="Amount of epochs")
    parser.add_argument('-b', '--batches', type=int,
                        default=16, help="Amount of batches")
    parser.add_argument('-s', '--size', type=int,
                        default=128, help="Size of batches")

    args = parser.parse_args()

    real_images = load_data(path=args.path, full_size=None)

    wgan = build_wgan()
    cbk = GANMonitor()
    wgan.fit(real_images, batch_size=args.size,
             epochs=args.epochs, callbacks=[cbk])

    tf.saved_model.save(wgan, './outputs/final_wgan')
