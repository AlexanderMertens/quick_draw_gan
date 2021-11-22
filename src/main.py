import numpy as np
from models.build_model import build_discriminator
from models.train_model import train_model
from visualization.visualise import plot_history


train_model(num_epochs=10, num_batch=4, batch_size=16)
