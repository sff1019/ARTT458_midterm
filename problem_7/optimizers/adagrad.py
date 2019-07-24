import numpy as np

from st_ops import st_ops


class AdaGrad(object):
    def __init__(self, lam, delta=0.01, lr=0.01):
        self.delta = delta
        self.lam = lam
        self.lr = lr
        self.grad_history = []

    def update(self, grad, params):
        self.grad_history.append(grad.flatten().tolist())
        ht = np.sqrt(np.sum(np.array(self.grad_history)
                            ** 2, axis=0).T) + self.delta
        ht = ht.reshape(2, 1)

        next_params = params - self.lr * (ht**-1 * grad)
        ht_inv = ht**-1
        params = np.array([
            st_ops(next_params[0], self.lam * self.lr * ht_inv[0]),
            st_ops(next_params[1], self.lam * self.lr * ht_inv[1])
        ])

        return params
