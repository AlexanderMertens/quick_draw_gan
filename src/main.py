import argparse
import os

from models.train_model import train_model
from scripts.download_data import download_data

if __name__ == "__main__":
    os.makedirs('./outputs/figures')

    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument('-p', '--path', type=str,
                        required=True, help="Path to location of data")
    parser.add_argument('--name', type=str, required=True,
                        help="Name of dataset")
    parser.add_argument('-e', '--epochs', type=int,
                        default=10, help="Amount of epochs")
    parser.add_argument('-b', '--batches', type=int,
                        default=16, help="Amount of batches")
    parser.add_argument('-s', '--size', type=int,
                        default=128, help="Size of batches")

    args = parser.parse_args()

    # Download data from azure
    download_data(args.path, args.name)

    # Train models, returns discriminator and generator at last epoch
    discriminator, generator = train_model(
        data_path='{folder}/{name}.npy'.format(
            folder=args.path, name=args.name),
        num_epochs=args.epochs,
        num_batch=args.batches,
        batch_size=args.size)

    # Save models at last epoch
    discriminator.save('./outputs/discriminator')
    generator.save('./outputs/generator')
