from data_help.data_transform import convert_to_image, flip_images_y
from data_help.make_dataset import simple_load_data
import data_help.data_constants as dc
from visualization.visualise import plot_images


dog = convert_to_image(simple_load_data(dc.TMP_DOG_DATA_PATH))
print(dog.shape)
plot_images(images=dog, path="figures/dogs.png")
plot_images(images=flip_images_y(dog[:100]), path="figures/flipped-dogs.png")
