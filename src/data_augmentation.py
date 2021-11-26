import numpy as np
import data_help.data_constants as dc
from data_help.data_transform import convert_to_image

from visualization.visualise import plot_images
from data_help.make_dataset import simple_load_data


full_size = 10000
images = simple_load_data(dc.ENVELOPE_DATA_PATH, full_size=full_size)
images = convert_to_image(images)
length = 100
proper_images = np.ones((1, 28, 28, 1))
for i in range(0, full_size, length):
    selection = images[i: i + length]
    plot_images(selection, numbering=range(0, length),
                path='figures/data_images.png', save=True, show=False)
    to_remove = []
    while True:
        x = input('Enter index to remove: >> ')
        if x == 'q':
            break
        to_remove.append(int(x))

    filtered_images = np.delete(selection, to_remove, axis=0)
    proper_images = np.append(proper_images, filtered_images, axis=0)

np.save(dc.FILTERED_ENVELOPE_DATA_PATH, proper_images[1:])
