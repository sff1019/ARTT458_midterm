import matplotlib.pyplot as plt
import numpy as np


LAMBDA = 2

np.random.seed(0)


def create_dataset(n):
    """
    Implemented dataset2
    """
    omega = np.random.randn()
    noise = 0.8 * np.random.randn(n)

    x = np.random.randn(n, 2) + 0
    y = 2 * (omega * x[:, 0] + x[:, 1] + noise > 0) - 1

    return x, y, omega


def calc_K(x_train, y_train, n):
    K = np.zeros([n, n])

    # calculate K = yi * yj * xi.T * xj
    for i in range(0, n):
        for j in range(0, n):
            K[i, j] = y_train[i] * y_train[j] * \
                np.dot(x_train[i].T, x_train[j])

    return K


def negative_dual(epochs, x_train, y_train):
    """
    Calculate the gradient of langrange
    """
    w = np.random.rand(2)
    losses = []

    for epoch in epochs:
        lr = 1 / (LAMBDA * epoch)
        grad = 0

        # Calculate the gradient of weight
        for index, x in enumerate(x_train):
            alpha = y_train[index] * np.dot(w.T, x)
            if alpha < 1:
                grad -= y_train[index] * x
            elif alpha == 1:
                grad -= 0
        grad += 2 * LAMBDA * w

        # update weight
        w = w - lr * grad

        # rerm loss
        loss = rerm(x_train, y_train, w)
        losses.append(loss)

    return w, losses


def score_dual_lagrange(epochs, x_train, y_train, n):
    """
    Calculate score of the dual Lagrange function
    """
    alpha = np.random.rand(n)
    K = calc_K(x_train, y_train, n)

    scores = []
    losses = []
    for epoch in epochs:
        lr = 1 / (LAMBDA * (epoch + 1))
        alpha -= lr * (1 / (2 * LAMBDA) * np.dot(K, alpha) - np.ones(n))
        alpha = np.clip(alpha, 0, 1)

        w_optimal = np.sum(
            (alpha * y_train).reshape([n, -1]) * x, axis=0) / (2 * LAMBDA)
        loss = rerm(x_train, y_train, w_optimal)
        losses.append(loss)

        score = - np.dot(alpha.T, np.dot(K, alpha)) / (4 * LAMBDA)
        score += np.dot(alpha.T, np.ones(n))
        scores.append(score)

    return scores, losses, w_optimal


def rerm(x_train, y_train, w):
    # calculate the regularized ERM
    _sum = 0

    for index, x in enumerate(x_train):
        _sum += max(0, 1 - y_train[index] * np.dot(w.T, x))
    loss = _sum + LAMBDA * np.dot(w.T, w)

    return loss


if __name__ == '__main__':
    epochs = range(1, 100)
    n = 400

    x, y, grad = create_dataset(n)

    # sum of hinge loss function and regularization
    w, losses = negative_dual(epochs, x, y)

    # score of dual lagrange and loss with optimal weights
    score, losses_optimal, w_optimal = score_dual_lagrange(epochs, x, y, 400)

    # Scatter graph
    plt.scatter(x[:, 0], x[:, 1], c=y)
    line_ideal = - grad * np.arange(-3, 4)
    line_original = - (w[0] / w[1]) * np.arange(-3, 4)
    line_dual = - (w_optimal[0] / w_optimal[1]) * np.arange(-3, 4)
    plt.title('Scatter Graph')
    plt.plot(list(range(-3, 4)), line_ideal, label='Ideal Line')
    plt.plot(list(range(-3, 4)), line_original, label='Original Problem')
    plt.plot(list(range(-3, 4)), line_dual, label='Dual Problem')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig('../../report/assets/prob_3_scatter.pdf')
    plt.clf()

    plt.title('Score of the dual Lagrange function and RERM')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Score and Loss', fontsize=11)
    plt.plot(list(range(1, len(losses) + 1)), losses, label='Loss')
    plt.plot(list(range(1, len(losses_optimal) + 1)),
             losses_optimal, label='Loss with optimal weight')
    plt.plot(list(range(1, len(score) + 1)), score, label='Score')
    plt.legend()
    plt.savefig('../../report/assets/prob_3_score_loss.pdf')
