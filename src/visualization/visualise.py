import matplotlib.pyplot as plt


def visualize_training_data(images, amount_rows=10, show=False, save=False):
    amount = amount_rows * 10
    plot_images(images, path="figures/training_data.png",
                amount_rows=amount_rows, show=show, save=save)


def plot_images(images, path=None, amount_rows=10, show=True, save=False, numbering=None):
    amount = amount_rows * 10
    fig = plt.figure(figsize=(amount_rows, 10))
    for i in range(amount):
        ax = fig.add_subplot(amount_rows, 10, i+1)
        ax.imshow(images[i], cmap='gray_r')
        ax.axis('off')
        if numbering is not None:
            ax.set_title(numbering[i])
    fig.tight_layout()
    if save:
        fig.savefig(path)
        plt.close(fig)

    if show:
        plt.show()


def plot_history(history, columns=['loss'], titles=['loss']):
    for i, column, title in zip(range(len(columns)), columns, titles):
        plt.subplot(len(columns), 1, i + 1)
        plt.title(title)
        plt.plot(history.history[column])
        plt.plot(history.history['val_{}'.format(column)])
    plt.show()


def plot_metrics(loss_array, title_array):
    fig = plt.figure()
    for loss, title, position in zip(loss_array, title_array, range(len(loss_array))):
        plot_metric(fig, loss, title, position + 1)
    fig.tight_layout()
    fig.savefig('./figures/results/metrics.png')
    plt.close(fig)


def plot_metric(figure, metric, name, rows=2, columns=1, position=1):
    ax = figure.add_subplot(rows, columns, position)
    ax.plot(metric)
    ax.set_title(name)
