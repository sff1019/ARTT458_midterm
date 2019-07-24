import cvxpy as cv
import matplotlib.pyplot as plt
import numpy as np

import apg
from trainer import Trainer
from optimizers.adadelta import Adadelta
from optimizers.adagrad import AdaGrad
from optimizers.adam import Adam
from optimizers.adamax import Adamax
from optimizers.nadam import Nadam
from optimizers.rmsprop import RMSprop
from optimizers.pg import PG


if __name__ == '__main__':
    A = np.array([[250, 15], [15,  4]])
    mu = np.array([[1], [2]])
    lam = 0.89

    # epochs = 1000
    epochs = 500
    # epochs = 100
    # lr = 0.1
    # lr = 0.01
    # lr = 0.001
    lr = 'optimal'

    # Figures for plotting
    loss_fig = plt.figure(1)  # loss fig is log scaled
    loss_ax = loss_fig.add_subplot(1, 1, 1)
    heat_map = plt.figure(2)
    heat_ax = heat_map.add_subplot(1, 1, 1)

    x_1 = np.arange(-1.5, 3, 0.01)
    x_2 = np.arange(-1.5, 3, 0.02)

    X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
    loss = np.zeros((len(x_1), len(x_2)))

    for i in range(len(x_1)):
        for j in range(len(x_2)):
            inr = np.vstack([x_1[i], x_2[j]])
            loss[i, j] = np.dot(np.dot((inr-mu).T, A), (inr - mu)) +\
                lam * (np.abs(x_1[i]) + np.abs(x_2[j]))
    # cvx
    w_lasso = cv.Variable((2, 1))
    obj_fn = cv.quad_form(w_lasso - mu, A) + lam * cv.norm(w_lasso, 1)
    objective = cv.Minimize(obj_fn)
    constraints = []
    prob = cv.Problem(objective, constraints)
    result = prob.solve(solver=cv.CVXOPT)
    w_lasso = w_lasso.value

    heat_ax.contour(X1, X2, loss, 80)
    heat_ax.plot(w_lasso[0], w_lasso[1], 'ko')

    min_loss = np.dot(np.dot((w_lasso - mu).T, A),
                      (w_lasso - mu)) + lam * np.sum(np.abs(w_lasso))

    # optimizers
    pg = PG(lam, A, None)
    adadelta = Adadelta((2, 1), lam)
    adagrad = AdaGrad(lam, 0.01, 0.01)
    adam = Adam(lam, 0.001)
    adamax = Adamax(lam, 0.002)
    nadam = Nadam(lam, 0.002)
    rmsprop = RMSprop(lam, 0.001)

    # trainers
    pg_trainer = Trainer(A, lam, mu, pg, epochs, 'Proximal Gradient')
    adadelta_trainer = Trainer(A, lam, mu, adadelta, epochs, 'Adadelta')
    adagrad_trainer = Trainer(A, lam, mu, adagrad, epochs, 'AdaGrad')
    adam_trainer = Trainer(A, lam, mu, adam, epochs, 'Adam')
    adamax_trainer = Trainer(A, lam, mu, adamax, epochs, 'Adamax')
    nadam_trainer = Trainer(A, lam, mu, nadam, epochs, 'Nadam')
    rmsprop_trainer = Trainer(A, lam, mu, rmsprop, epochs, 'RMSprop')

    # run
    apg_losses = apg.run(mu, lam)
    pg_losses = pg_trainer.train()
    adadelta_params, adadelta_losses = adadelta_trainer.train()
    adagrad_params, adagrad_losses = adagrad_trainer.train()
    adam_params, adam_losses = adam_trainer.train()
    adamax_params, adamax_losses = adamax_trainer.train()
    nadam_params, nadam_losses = nadam_trainer.train()
    rmsprop_params, rmsprop_losses = rmsprop_trainer.train()

    # calc minimum loss
    min_of_min = np.min([
        min_loss,
        np.min(adadelta_losses),
        np.min(adagrad_losses),
        np.min(adam_losses),
        np.min(adamax_losses),
        np.min(nadam_losses),
        np.min(rmsprop_losses),
        np.min(apg_losses)
    ])

    # plot
    # pg_trainer.plot(heat_ax, loss_ax, min_of_min, False, True)
    adadelta_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)
    adagrad_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)
    adam_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)
    adamax_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)
    nadam_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)
    rmsprop_trainer.plot(heat_ax, loss_ax, min_of_min, True, True)

    # show legend
    heat_ax.legend()
    loss_ax.legend()

    # add x, y labels and titles to figures
    heat_ax.set_title(
        f'Parameter Transition on Epochs: {epochs}, Learning Rate: {lr}'
    )
    heat_ax.set_xlabel('x1')
    heat_ax.set_ylabel('x2')

    loss_ax.set_title(
        f'Loss Transition on Epochs: {epochs}, Learning Rate: {lr}'
    )
    loss_ax.set_xlabel('# of iteration')
    loss_ax.set_ylabel('J(w^t) - J(w^hat)')

    plt.xlim(-1.5, 3)
    plt.ylim(-1.5, 3)
    str_lr = str(lr).replace('.', '')
    heat_map.savefig(f'../../report/assets/prob_7_param_{epochs}_{str_lr}.pdf')
    loss_fig.savefig(f'../../report/assets/prob_7_loss_{epochs}_{str_lr}.pdf')
