import numpy as np


# proximal gradient
def st_ops(mu, q):
    x_proj = np.zeros(mu.shape)
    for i in range(len(mu)):
        if mu[i] > q:
            x_proj[i] = mu[i] - q
        else:
            if np.abs(mu[i]) < q:
                x_proj[i] = 0
            else:
                x_proj[i] = mu[i] + q
    return x_proj
