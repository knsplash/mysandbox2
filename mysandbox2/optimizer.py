from typing import Optional
import datetime

import numpy as np
import pandas as pd
from dask.distributed import Client

from scipy.optimize import minimize, OptimizeResult

import optuna
from optuna import Trial
from optuna.samplers import BaseSampler

from .calculator import AbstractCalculator, HyperSphere
from .core import History


class AbstractOptimizer:

    def __init__(self, calculator: Optional[AbstractCalculator] = None):
        self.parameters: pd.DataFrame = pd.DataFrame()
        self.objectives: dict = dict()
        self.client: Optional[Client] = None
        self.calculator: Optional[AbstractCalculator] = calculator or HyperSphere()
        self.history: Optional[History] = None
        self.n_trials: Optional[int] = None

    def setup(self):
        pass

    def main(self):
        raise NotImplementedError()

    def finalize(self):
        pass

    def optimize(self, n_trials: Optional[int] = None):
        self.n_trials = n_trials
        self.history = History(self.parameters, self.objectives)
        self.setup()
        self.main()
        self.finalize()


class ScipyOptimizer(AbstractOptimizer):

    def __init__(
            self,
            method: Optional[str] = None,
            calculator: Optional[AbstractCalculator] = None
    ):
        self.method = method
        self._current_trial = 0
        super().__init__(calculator)

    def _objective(self, x: np.ndarray) -> float:

        # check
        self._current_trial += 1
        if self._current_trial > (self.n_trials or np.inf):
            raise StopIteration()

        # update calculator
        self.calculator.calculate(x)

        # calc y
        obj_fun = list(self.objectives.values())[0]
        y: float = obj_fun(self.calculator)

        # rec
        self.history.record(x, np.array((y,)))

        return y

    def main(self):
        assert len(self.objectives) == 1
        try:
            minimize(
                fun=self._objective,
                x0=self.parameters.init.values,
                options=None,
                bounds=self.parameters[['lb', 'ub']].values,
                method=self.method,
                tol=0.01,
            )
        except StopIteration:
            pass


class OptunaOptimizer(AbstractOptimizer):

    sampler = None
    storage = None
    study_name = 'my-study'
    study = None

    def __init__(
            self,
            sampler: Optional[BaseSampler] = None,
            calculator: Optional[AbstractCalculator] = None
    ):
        self.sampler = sampler
        super().__init__(calculator)

    def setup(self):
        # self.storage = optuna.integration.DaskStorage(
        #     storage=None,
        #     name=None,
        #     client=self.client,
        # )
        self.storage = f'sqlite:///{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.db'

        self.study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            study_name=self.study_name,
            directions=['minimize']*len(self.objectives),
            load_if_exists=True,
        )

    def main(self):
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=None,
        )

        study.optimize(
            func=self._objective,
            n_trials=self.n_trials,
            timeout=None,
            callbacks=None,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    def _objective(self, trial: Trial):
        # create x
        x = []
        for i, row in self.parameters.iterrows():
            name = row['name']
            lb = row['lb']
            ub = row['ub']
            x.append(trial.suggest_float(name, low=lb, high=ub, step=None))
        x = np.array(x)

        # update calculator
        self.calculator.calculate(x)

        # calc y
        y = []
        for obj_name, obj_fun in self.objectives.items():
            y.append(obj_fun(self.calculator))
        y = np.array(y)

        # rec
        self.history.record(x, y)

        return tuple(y)
