import numpy as np

from st_ops import st_ops


class PG(object):
    def __init__(self, lam, A, lr):
        self.lam = lam
        if lr is None:
            self.lr = 1.01 * np.max(np.linalg.eig(2 * A)[0])
        else:
            self.lr = lr

    def update(self, grad, params):

        next_params = params - 1/self.lr * grad
        params = st_ops(next_params, self.lam * 1 / self.lr)

        return params
