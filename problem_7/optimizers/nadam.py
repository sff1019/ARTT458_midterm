"""
References

[1] Timothy Dozat. 2015. Incorporating Nesterov Momentum into Adam

"""
import numpy as np

from st_ops import st_ops


class Nadam(object):
    def __init__(self, lam, lr=0.01, beta_1=0.9, beta_2=0.99999, epsilon=1e-8):
        self.lam = lam
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.mu = beta_1 * (1 - 0.5 * (0.96**(1/250)))
        self.mu_cache = self.mu
        self.mu_prod = self.mu_cache

        self.m = np.zeros((2, 1))
        self.v = np.zeros((2, 1))
        self.iter = 0

    def update(self, grad, params):
        self.iter += 1
        grad_hat = grad / (1 - self.mu_prod)

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad * grad)

        self.mu_cache = self.mu
        self.mu = self.beta_1 * (1 - 0.5 * (0.96**((self.iter+1) / 250)))
        self.mu_prod *= self.mu

        m_hat = self.m / (1 - self.mu_prod)
        v_hat = self.v / (1 - self.beta_2**self.iter)
        m_bar = (1 - self.mu_cache) * grad_hat + self.mu * m_hat

        update_rate = self.lr * \
            np.ones_like(params) / (np.sqrt(v_hat) + self.epsilon)
        next_params = params - m_bar * update_rate

        params = np.array([
            st_ops(next_params[0], self.lam * update_rate[0]),
            st_ops(next_params[1], self.lam * update_rate[1])
        ])

        return params
