import matplotlib.pyplot as plt
from data_help.make_dataset import load_dogs


def visualize_training_data(amount_rows=10, show=False, save=False):
    amount = amount_rows * 10
    images = load_dogs(full_size=amount, verbose=True).reshape(amount, 28, 28)

    plt.figure(figsize=(amount_rows, 10))
    for i in range(amount):
        plt.subplot(amount_rows, 10, i+1)
        plt.imshow(images[i], cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig("figures/training_data.png")

    if show:
        plt.show()
