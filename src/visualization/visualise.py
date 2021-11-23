import matplotlib.pyplot as plt


def visualize_training_data(images, amount_rows=10, show=False, save=False):
    amount = amount_rows * 10
    plot_images(images, path="figures/training_data.png",
                amount_rows=amount_rows, show=show, save=save)


def plot_images(images, path=None, amount_rows=10, show=True, save=False):
    amount = amount_rows * 10
    fig = plt.figure(figsize=(amount_rows, 10))
    for i in range(amount):
        ax = fig.add_subplot(amount_rows, 10, i+1)
        ax.imshow(images[i], cmap='gray_r')
        ax.axis('off')
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


def plot_metrics(d_loss_avg, d_accuracy_avg, g_loss_avg, g_accuracy_avg):
    fig = plt.figure()
    plot_metric(fig, d_loss_avg, 'Discriminator loss', position=1)
    plot_metric(fig, d_accuracy_avg, 'Discriminator accuracy', position=2)
    plot_metric(fig, g_loss_avg, 'GAN loss', position=3)
    plot_metric(fig, g_accuracy_avg, 'GAN accuracy', position=4)
    fig.savefig('./figures/results/metrics.png')
    plt.close(fig)


def plot_metric(figure, metric, name, rows=2, columns=2, position=1):
    ax = figure.add_subplot(rows, columns, position)
    ax.plot(metric)
    ax.set_title(name)
