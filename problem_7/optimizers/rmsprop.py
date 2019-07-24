"""
References

[1] Geoffrey Hinton, Nitish Srivastava, Kevin Swersky. 2012. Lecture 6a "Overview of mini-batch gradient"
[2] Vitaly Bushaev (2018, Sep 3). Understanding RMSprop — faster neural network learning.
https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a

"""
import numpy as np

from st_ops import st_ops


class RMSprop(object):
    def __init__(self, lam, lr=0.01, beta=0.9, epsilon=1e-8):
        self.lam = lam
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        self.e = np.zeros((2, 1))

    def update(self, grad, params):
        self.e = self.beta * self.e + (1 - self.beta) * (grad * grad)
        update_rate = 1 / np.sqrt(self.e + self.epsilon)
        update_rate_inv = self.e**-1

        next_params = params - self.lr * grad * update_rate
        params = np.array([
            st_ops(next_params[0], self.lam * self.lr * update_rate_inv[0]),
            st_ops(next_params[1], self.lam * self.lr * update_rate_inv[1]),
        ])

        return params
