import pickle

import numpy as np
import pandas as pd

from mysandbox2.optimizer import ScipyOptimizer, AbstractCalculator


def distance_to_minus(hs):
    d = len(hs.x)
    return np.linalg.norm(hs.x - np.sqrt(1./d)*np.ones(d))


if __name__ == '__main__':

    method = 'trust-constr'

    opt = ScipyOptimizer(method=method)

    opt.parameters = pd.DataFrame(
        dict(
            name=['r', 'fai0', 'fai1'],
            init=[0.5, 0, 0],
            lb=[0, 0, 0],
            ub=[1, np.pi, 2*np.pi],
        )
    )

    opt.objectives = dict(
        distance=distance_to_minus,
    )

    opt.optimize(None)

    print(opt.history.df)

    with open(f'history-{method}.pkl', 'wb') as f:
        pickle.dump(opt.history.df, f)
