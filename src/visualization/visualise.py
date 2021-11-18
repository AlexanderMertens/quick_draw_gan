import matplotlib.pyplot as plt


def visualize_training_data(images, amount_rows=10, show=False, save=False):
    amount = amount_rows * 10
    plot_images(images, path="figures/training_data.png",
                amount_rows=amount_rows, show=show, save=save)


def plot_images(images, path=None, amount_rows=10, show=True, save=False):
    amount = amount_rows * 10
    plt.figure(figsize=(amount_rows, 10))
    for i in range(amount):
        plt.subplot(amount_rows, 10, i+1)
        plt.imshow(images[i], cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(path)

    if show:
        plt.show()


def plot_history(history, columns=['loss'], titles=['loss']):
    for i, column, title in zip(range(len(columns)), columns, titles):
        plt.subplot(len(columns), 1, i + 1)
        plt.title(title)
        plt.plot(history.history[column])
        plt.plot(history.history['val_{}'.format(column)])
    plt.show()
