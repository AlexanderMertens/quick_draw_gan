"""
Script to aid filtering images manually.
"""
import numpy as np
import data_help.data_constants as dc
from data_help.data_transform import convert_to_image

from visualization.visualise import plot_images
from data_help.make_dataset import simple_load_data

full_size = 10000
images = simple_load_data(dc.DOG_DATA_PATH,
                          start=2*full_size,)
images = convert_to_image(images)
length = 100
proper_images = simple_load_data(
    dc.FILTERED_BUTTERFLY_DATA_PATH_OLD, end=full_size)
print(proper_images.shape)
for i in range(0, full_size, length):
    selection = images[i: i + length]
    plot_images(selection, numbering=range(0, length),
                path='figures/data_images.png', show=False)
    to_remove = []
    while True:
        x = input('Enter index to remove: >> ')
        if not x.isdigit() or int(x) > 100:
            break
        to_remove.append(int(x))

    filtered_images = np.delete(selection, to_remove, axis=0)
    proper_images = np.append(proper_images, filtered_images, axis=0)
    if (i + length) % (10 * length) == 0:
        print('Progress saved')
        np.save('{}_{}.npy'.format(dc.FILTERED_BUTTERFLY_DATA_PATH, i),
                proper_images[1:])
np.save(dc.FILTERED_BUTTERFLY_DATA_PATH, proper_images[:])
quit()
