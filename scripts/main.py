import pickle

import numpy as np
import pandas as pd
from optuna.integration import BoTorchSampler
from optuna.samplers import MOTPESampler

from mysandbox2.optimizer import OptunaOptimizer, AbstractCalculator


def x(hs):
    return hs.x[0]


def y(hs):
    return hs.x[1]


def z(hs):
    return hs.x[2]


if __name__ == '__main__':

    # sampler = BoTorchSampler()
    sampler = MOTPESampler()
    # sampler = None

    opt = OptunaOptimizer(sampler=sampler)

    opt.parameters = pd.DataFrame(
        dict(
            name=['r', 'fai0', 'fai1'],
            init=[0.5, 0, 0],
            lb=[0, 0, 0],
            ub=[1, np.pi, 2*np.pi],
        )
    )

    opt.objectives = dict(
        x=x,
        y=y,
        z=z,
    )

    opt.optimize(100)

    print(opt.history.df)

    with open('history-100-MOTPE.pkl', 'wb') as f:
        pickle.dump(opt.history.df, f)
