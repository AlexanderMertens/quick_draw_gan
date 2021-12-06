import argparse
import logging
import tensorflow as tf

from models.train_model import train_model

if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument('-e', '--epochs', type=int,
                        default=10, help="Amount of epochs")
    parser.add_argument('-b', '--batches', type=int,
                        default=16, help="Amount of batches")
    parser.add_argument('-s', '--size', type=int,
                        default=128, help="Size of batches")

    args = parser.parse_args()

    gan = train_model(num_epochs=args.epochs, num_batch=args.batches,
                      batch_size=args.size)
