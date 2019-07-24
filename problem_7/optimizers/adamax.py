"""
References

[1] Diederik P. Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization. 2014

"""
import numpy as np

from st_ops import st_ops


class Adamax(object):
    def __init__(self, lam, lr=0.01, beta_1=0.9, beta_2=0.99999):
        self.lam = lam
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.m = np.zeros((2, 1))
        self.v = np.zeros((2, 1))
        self.iter = 0

    def update(self, grad, params):
        self.iter += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        m_hat = self.m / (1 - self.beta_1**self.iter)

        self.v = np.maximum(self.beta_2 * self.v, np.abs(grad))

        update_rate = self.lr * np.ones_like(params) / self.v
        next_params = params - m_hat * update_rate

        params = np.array([
            st_ops(next_params[0], self.lam * update_rate[0]),
            st_ops(next_params[1], self.lam * update_rate[1])
        ])

        return params
