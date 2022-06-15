#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix

from nda.experiment_utils import run_exp

if __name__ == '__main__':
    n_agents = [20]
    m = 5000
    dim = 40

    kappa = 5000
    mu = 5e-6

    n_iters = 20

    for n_agent in n_agents:
        p = LogisticRegression(n_agent=n_agent, m=m, dim=dim, noise_ratio=0.05, graph_type='er', kappa=kappa, graph_params=0.4)
        print(p.n_edges)

        x_0 = np.random.rand(dim, n_agent)
        x_0_mean = x_0.mean(axis=1)
        W, alpha = generate_mixing_matrix(p)
        print('alpha = ' + str(alpha))


        # eta = 2 / (p.L + p.sigma)
        eta = 10e-3
        n_inner_iters = int(m * 0.05)
        batch_size = int(m / 10)
        n_dgd_iters = n_iters * 20
        n_svrg_iters = n_iters * 20
        n_dsgd_iters = int(n_iters * m / batch_size)


        distributed = [
            DGD_tracking(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W),
            EXTRA(p, n_iters=n_dgd_iters, eta=eta/2, x_0=x_0, W=W),
            NIDS(p, n_iters=n_dgd_iters, eta=eta, x_0=x_0, W=W),

            ADMM(p, n_iters=n_iters, rho=1, x_0=x_0_mean)
            ]

        res = run_exp(distributed, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_cpu_processes=4, save=True)
        plt.title(f"N_AGENT: {n_agent}")
        plt.show()
