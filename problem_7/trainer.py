import numpy as np


class Trainer(object):
    def __init__(self, A, lam, mu, optimizer, epochs, name=''):
        self.A = A
        self.lam = lam
        self.mu = mu
        self.optimizer = optimizer
        self.epochs = epochs
        self.name = name

        self.params = np.array([[3], [-1]])

        self.params_history = []
        self.loss_list = []

    def train(self):
        for t in range(self.epochs):
            self.params_history.append(self.params.T)
            grad = 2 * np.dot(self.A, self.params - self.mu)

            self.params = self.optimizer.update(grad, self.params)

            loss = np.dot(np.dot((self.params - self.mu).T, self.A), (self.params - self.mu)) + \
                self.lam * (np.abs(self.params[0]) + np.abs(self.params[1]))
            self.loss_list.append(loss)

        self.params_history = np.vstack(self.params_history)
        self.loss_list = np.vstack(self.loss_list)

        return self.params_history, self.loss_list

    def plot(self, contour_fig, loss_fig=None, min_of_min=None, contour=True, loss=False):
        """
        Plot Contour map and Loss fig

        Return: contour_fig, loss_fig
        """
        if contour:
            self.plot_contour(contour_fig)

        if loss_fig is not None and loss:
            self.plot_loss(loss_fig, min_of_min)

        return contour_fig, loss_fig

    def plot_contour(self, fig):
        fig.plot(
            self.params_history[:, 0],
            self.params_history[:, 1],
            'o-',
            markersize=3,
            linewidth=0.5,
            label=self.name,
        )

        return fig

    def plot_loss(self, fig, min_of_min):
        fig.semilogy(self.loss_list-min_of_min, 's-',
                     markersize=1, linewidth=0.5, label=self.name)

        return fig
