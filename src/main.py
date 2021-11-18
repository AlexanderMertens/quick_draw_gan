import numpy as np
from data_help.make_dataset import CAT_DATA_PATH, DOG_DATA_PATH, load_data, split_data
from models.build_model import build_cats_and_dogs_discriminator
from visualization.visualise import plot_images, visualize_training_data


size = 5000
cats_data = load_data(path=CAT_DATA_PATH, full_size=size)
cats_y = np.zeros(size)
dogs_data = load_data(path=DOG_DATA_PATH, full_size=size)
dogs_y = np.ones(size)
data = np.concatenate((cats_data, dogs_data))
Y = np.concatenate((cats_y, dogs_y))
x_train, y_train, x_test, y_test = split_data(data, Y)
plot_images(x_train)
model = build_cats_and_dogs_discriminator()
