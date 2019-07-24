"""
References

[1] Matthew D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. 2012

"""
import numpy as np

from st_ops import st_ops


class Adadelta(object):
    def __init__(self, param_size, lam, rho=0.95, epsilon=1e-6):
        self.rho = 0.95
        self.epsilon = 1e-6
        self.lam = lam

        self.acc_grads = np.zeros(param_size)
        self.acc_updates = np.zeros(param_size)

    def update(self, grad, params):
        self.acc_grads = self.rho * self.acc_grads + \
            (1 - self.rho) * grad * grad
        rms_grads = np.sqrt(self.acc_grads + self.epsilon)
        rms_updates = np.sqrt(self.acc_updates + self.epsilon)

        update_rate = - rms_updates / rms_grads
        self.acc_updates = self.rho * self.acc_updates + \
            (1 - self.rho) * update_rate * grad * update_rate * grad

        next_params = params + update_rate * grad

        params = np.array([
            st_ops(next_params[0], self.lam * update_rate[0]),
            st_ops(next_params[1], self.lam * update_rate[1])
        ])

        return params
