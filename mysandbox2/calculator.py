import numpy as np


class AbstractCalculator:

    def calculate(self, x):
        raise NotImplementedError()


class HyperSphere(AbstractCalculator):

    def __init__(self, n=3):
        assert n >= 2
        self.n = n
        self.r = 0.
        self.fai = np.zeros(n-1)
        self.x = np.zeros(self.n)

    def _compute(self):
        tmp_x = np.empty(self.n)
        _x = self.r
        for i in range(self.n-1):
            tmp_x[i] = _x * np.cos(self.fai[i])
            _x *= np.sin(self.fai[i])
        tmp_x[self.n-1] = _x
        self.x = tmp_x

    def calculate(self, x):
        r, *fai = x
        self.r = r
        self.fai = np.array(fai)
        self._compute()
