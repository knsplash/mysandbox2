import pickle

import numpy as np

# https://botorch.org/#quickstart
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

# https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
import gpytorch

import plotly.express as px


if __name__ == '__main__':

    # path = 'history-100-None.pkl'
    # path = 'history-100-BoTorch.pkl'
    path = 'history-100-MOTPE.pkl'

    with open(path, 'rb') as f:
        df = pickle.load(f)
    px.scatter_matrix(
        df,
        dimensions=[c for c in df.columns if 'obj' in c]
    ).show()

    x = df[[c for c in df.columns if 'prm' in c]].values
    y = df[[c for c in df.columns if 'obj' in c]].values

    train_x = torch.from_numpy(x.astype(float)).clone()
    train_y = torch.from_numpy(y.astype(float)).clone()

    # https://botorch.org/#quickstart
    # train_y = standardize(train_y)
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    test_x = torch.from_numpy(
        np.array(
            [
                [0, 0, 0],
                [.2, 0, 0],
                [.4, 0, 0],
                [.6, 0, 0],
                [.8, 0, 0],
                [1, 0, 0],
            ]
        ).astype(float)
    )
    # gp.mean_module(test_x)  # 何???
    f_preds = gp(test_x)
    y_preds = gp.likelihood(gp(test_x))  # 合ってる？？

    likelihood = gp.likelihood  # 合ってる？
    model = gp

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    # f_samples = f_preds.sample(sample_shape=torch.Size(1000, ))  # error


    # https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    import matplotlib.pyplot as plt
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        # Plot training data as black stars
        ax.plot(train_x.numpy()[:,0], train_y.numpy()[:,0], 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy()[:,0], observed_pred.mean.numpy()[0], 'b')
        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        # ax.legend(['Observed Data', 'Mean', 'Confidence'])

        # plt.show()