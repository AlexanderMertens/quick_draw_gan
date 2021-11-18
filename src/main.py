from matplotlib import pyplot as plt
import numpy as np
from data_help.make_dataset import CAT_DATA_PATH, DOG_DATA_PATH, create_cats_and_dogs_data, load_data, split_data
from models.build_model import build_cats_and_dogs_discriminator
from visualization.visualise import plot_history, plot_images, visualize_training_data


size = 5000
x_train, x_test, y_train, y_test = create_cats_and_dogs_data(size)

model = build_cats_and_dogs_discriminator()
model.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history = model.fit(x=x_train, y=y_train, batch_size=500,
                    epochs=50, validation_data=(x_test, y_test))

plot_history(history, columns=['loss', 'binary_accuracy'], titles=[
             'loss', 'accuracy'])
