import numpy as np


def sgd(params_init, A, mu, st_ops, lam):
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])
    L /= 2

    params_history = []
    params = params_init

    for t in range(100):
        params_history.append(params.T)
        grad = 2 * np.dot(A, params-mu)
        next_params = params - 1/L * grad
        params = st_ops(next_params, lam * 1 / L)

    params_history = np.vstack(params_history)

    return params_history
