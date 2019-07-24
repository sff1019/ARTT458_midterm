import numpy as np

from st_ops import st_ops


def run(mu, lam):
    A = np.array([[300, 0.5], [0.5,   1]])
    x_init = np.array([[3], [-1]])
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])
    xt = x_init
    x_historyApg = [x_init.T]
    s_history = [1]
    v_history = [x_init.T]
    fvaluesApg = []

    for t in range(170):
        vt = v_history[-1].T
        grad = 2 * np.dot(A, vt - mu)
        xth = vt - 1/L * grad
        xt = st_ops(xth, lam * 1 / L)
        x_historyApg.append(xt.T)

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + \
            lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvaluesApg.append(fv)

        sprev = s_history[-1]
        st = (1 + np.sqrt(1 + 4 * sprev**2)) / 2.0
        s_history.append(st)
        qt = (sprev - 1.0) / st
        vnew = xt + np.dot(qt, (xt - x_historyApg[-2].T))
        v_history.append(vnew.T)
    x_historyApg = np.vstack(x_historyApg)
    fvaluesApg = np.vstack(fvaluesApg)
    s_history = np.vstack(s_history)
    v_history = np.vstack(v_history)

    return fvaluesApg
