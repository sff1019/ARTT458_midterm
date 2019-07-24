###################################################################
# Implementation of SVG in Lagulange problems (original and dual) #
###################################################################

import random

import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 2.
N = 400
fig = plt.figure(figsize=(20, 16))
loss_graph = fig.add_subplot(2, 2, 1)
posi_graph = fig.add_subplot(2, 2, 2)
loss_posi_graph = fig.add_subplot(2, 2, 4)
scatter_graph = fig.add_subplot(2, 2, 3)
RANGE = range(1, 51)


def gen_dataset():
    omega = random.random()
    noise = 0.8 * np.random.randn(N, 1)
    x = np.random.randn(N, 2)
    y = 2 * (omega * x[:, 0] + x[:, 1] + noise[:, 0] > 0) - 1
    return x, y, omega


def eval_negative(x_train, y_train, w):
    s = 0.
    for t, x in enumerate(x_train):
        alpha = y_train[t] * np.dot(w.T, x)
        if alpha <= 1:
            s += 1 - alpha
    s += LAMBDA * np.linalg.norm(w)
    return s


def negative_dual(x_train, y_train):
    w = np.random.rand(2)
    loss = []
    for epoch in RANGE:
        lr = 1 / (LAMBDA * epoch)
        diff = 0.
        for t, x in enumerate(x_train):
            alpha = y_train[t] * np.dot(w.T, x)
            if alpha < 1:
                diff -= y_train[t] * x
            elif alpha == 1:
                diff -= lr * y_train[t] * x
        diff += 2 * LAMBDA * w
        w = w - lr * diff

        eval_loss = eval_negative(x_train, y_train, w)
        loss.append(eval_loss)

    loss_graph.plot(list(range(1, len(loss) + 1)), loss)
    loss_posi_graph.plot(list(range(1, len(loss) + 1)), loss, label='Loss')
    loss_posi_graph.legend()
    return w


def calc_weight_from_alpha(x, y, alpha):
    posi_sum = np.sum((alpha * y).reshape([N, -1]) * x, axis=0)
    w = (1 / (2 * LAMBDA)) * posi_sum
    return w


def eval_positive(alpha, K):
    # dual lagurange function
    s = (- 1 / (4 * LAMBDA)) * np.dot(alpha, np.dot(K, alpha))
    s += np.dot(alpha, np.ones(N))
    return s


def positive_dual(x_train, y_train):
    alpha = np.random.rand(N)
    K = np.zeros([N, N])

    for i in range(0, N):
        for j in range(0, N):
            K[i, j] = y_train[i] * y_train[j] * \
                np.dot(x_train[i], x_train[j])

    loss = []
    posi_loss = []
    for epoch in RANGE:
        lr = 1 / (LAMBDA * (epoch + 1))
        alpha = alpha - lr * (1 / (2 * LAMBDA) * np.dot(K, alpha) - np.ones(N))
        alpha = np.clip(alpha, 0, 1)

        eval_loss = eval_positive(alpha, K)
        w_posi = calc_weight_from_alpha(x_train, y_train, alpha)
        eval_posi_loss = eval_negative(x_train, y_train, w_posi)
        loss.append(eval_loss)
        posi_loss.append(eval_posi_loss)
    posi_graph.plot(list(range(1, len(loss) + 1)), loss)
    loss_posi_graph.plot(list(range(1, len(loss) + 1)), loss, label='Score')
    loss_posi_graph.plot(list(range(1, len(posi_loss) + 1)),
                         posi_loss, label='Loss (Calculated via Dual Problem)')
    loss_posi_graph.legend()
    return alpha


if __name__ == '__main__':
    x, y, grad = gen_dataset()
    scatter_graph.scatter(x[:, 0], x[:, 1], c=y)
    loss_posi_graph.set_xlabel('Epoch', fontsize=11)
    loss_posi_graph.set_ylabel('Score & Loss', fontsize=11)

    line = - grad * np.arange(-3, 4)
    scatter_graph.plot(list(range(-3, 4)), line, label='Ideal Line')
    scatter_graph.legend()

    w = negative_dual(x, y)
    loss_graph.set_xlabel('Epoch', fontsize=11)
    loss_graph.set_ylabel('loss', fontsize=11)
    line = - (w[0] / w[1]) * np.arange(-3, 4)
    scatter_graph.plot(list(range(-3, 4)), line, label='Original Problem')
    scatter_graph.legend()

    alpha = positive_dual(x, y)
    posi_graph.set_xlabel('Epoch', fontsize=11)
    posi_graph.set_ylabel('Score', fontsize=11)

    w_posi = calc_weight_from_alpha(x, y, alpha)
    line = - (w_posi[0] / w_posi[1]) * np.arange(-3, 4)
    scatter_graph.plot(list(range(-3, 4)), line, label='Dual Problem')
    scatter_graph.legend()

    scatter_graph.set_xlabel('x1', fontsize=11)
    scatter_graph.set_ylabel('x2', fontsize=11)
    plt.show()
