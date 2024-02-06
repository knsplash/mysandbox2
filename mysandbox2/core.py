import numpy as np
import pandas as pd


class History:

    df: pd.DataFrame = pd.DataFrame()

    def __init__(
            self,
            parameters: pd.DataFrame,
            objectives: dict,
    ):
        columns = []
        columns.extend([f'prm_{name}' for name in parameters['name'].values])
        columns.extend([f'obj_{name}' for name in objectives.keys()])
        self.df = pd.DataFrame(columns=columns)

    def record(
            self,
            x: np.ndarray,  # prm
            y: np.ndarray  # obj
    ):
        row = []
        row.extend(list(x))
        row.extend(list(y))
        self.df.loc[len(self.df)] = row
