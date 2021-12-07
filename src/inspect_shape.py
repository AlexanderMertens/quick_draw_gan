from data_help.make_dataset import simple_load_data
import data_help.data_constants as dc


dog = simple_load_data(dc.TMP_DOG_DATA_PATH)
print(dog.shape)
