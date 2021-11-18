from matplotlib import pyplot as plt
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
x_train, x_test, y_train, y_test = split_data(data, Y)

print(x_train.shape)
print(y_train.shape)
model = build_cats_and_dogs_discriminator()
model.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history = model.fit(x=x_train, y=y_train, batch_size=500,
                    epochs=50, validation_data=(x_test, y_test))

plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')

plt.subplot(2, 1, 2)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('accuracy')
plt.show()
