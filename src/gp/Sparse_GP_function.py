#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
import tqdm.notebook as tqdm
import torch
import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_number):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        kernel = gpytorch.kernels.RBFKernel()
        self.base_covar_module = ScaleKernel(kernel)

        # Randomly select Inducing_number points from train_x
        torch.manual_seed(42)  # Fix the random seed to ensure reproducibility
        selected_indices = torch.randperm(train_x.size(0))[:inducing_number]  # Generate random indices
        inducing_points = train_x[selected_indices]  # Select inducing points from train_x

        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def sparse_gp_model_train(x, y, Inducing_number, training_iterations, predic_warmup=True, train_viz_disable=False):

    X_torch = torch.from_numpy(x).float()
    Y_torch = torch.from_numpy(y).float()
    Y_torch = Y_torch.squeeze()  # Ensure train_y has shape [N]

    # Normalize the data
    X_mean, X_std = X_torch.mean(0), X_torch.std(0)
    X_torch = (X_torch - X_mean) / X_std

    Y_min, Y_max = Y_torch.min(), Y_torch.max()
    
    if Y_min == Y_max:  # All Y values are the same
        # print("Warning: All Y values are the same. Normalization skipped.")
        Y_torch = torch.zeros_like(Y_torch)  # Set normalized Y to 0
    else:
        Y_torch = 2 * ((Y_torch - Y_min) / (Y_max - Y_min)) - 1

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(X_torch, Y_torch, likelihood, Inducing_number)
    # save the scaler for prediction
    model.X_mean = X_mean
    model.X_std = X_std
    model.Y_min = Y_min
    model.Y_max = Y_max
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    iterator = tqdm.tqdm(range(training_iterations), desc="Train", disable=train_viz_disable)

    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X_torch)
        # Calc loss and backprop derivatives
        loss = -mll(output, Y_torch)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

    # predict warm-up
    if predic_warmup:
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_torch[:1]))

    return model, likelihood

def sparse_gp_model_predict(model, likelihood, xtest):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    xtest_torch = torch.from_numpy(xtest+0.001).float() # TODO there is bug in gpytorch, if xtext is close to the training data, it will raise error
    xtest_torch = (xtest_torch - model.X_mean) / model.X_std
    noise = 1e-5 * torch.randn_like(xtest_torch)
    xtest_torch = xtest_torch + noise

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(xtest_torch))

        Gpytorch_SVGP_mean = observed_pred.mean.numpy()
        Gpytorch_SVGP_var = observed_pred.variance.sqrt().detach().cpu().numpy()
        Gpytorch_SVGP_inducing_points = model.covar_module.inducing_points.detach().cpu().numpy()

        # # Restore the mean to the original scale
        Gpytorch_SVGP_mean = (Gpytorch_SVGP_mean + 1) * (model.Y_max.numpy() - model.Y_min.numpy()) / 2 + model.Y_min.numpy()
        Gpytorch_SVGP_var = Gpytorch_SVGP_var * (model.Y_max.numpy() - model.Y_min.numpy()) / 2
        Gpytorch_SVGP_inducing_points = Gpytorch_SVGP_inducing_points * model.X_std.numpy() + model.X_mean.numpy()

        return Gpytorch_SVGP_mean, Gpytorch_SVGP_var, Gpytorch_SVGP_inducing_points

def sparse_gp_model_predict_fast(model, likelihood, xtest):
    """_summary_
    only calculate the mean
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    xtest_torch = torch.from_numpy(xtest).float()
    xtest_torch = (xtest_torch - model.X_mean) / model.X_std

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(xtest_torch))

        Gpytorch_SVGP_mean = observed_pred.mean.numpy()

        # # Restore the mean to the original scale
        Gpytorch_SVGP_mean = (Gpytorch_SVGP_mean + 1) * (model.Y_max.numpy() - model.Y_min.numpy()) / 2 + model.Y_min.numpy()

        return Gpytorch_SVGP_mean


def sparse_gp_plot(xtrain, ytrain, xtest, predictive_mean, predictive_std, inducing_point, ax=None):
    """
    Plot the sparse GP model with observations, latent function, predictive mean, confidence intervals, and inducing points.

    GP Parameters:
    - xtrain: Training data input
    - ytrain: Training data output
    - xtest: Test data input
    - predictive_mean: Predictive mean of the GP model
    - predictive_std: Predictive standard deviation of the GP model
    - inducing_points: Inducing points used in the sparse GP
    - ax: Axis object for plotting (optional)

    Returns:
    - ax: The axis object with the plot
    """
    # Plotting

    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if ax is None:
        ax = plt.gca()  # If no ax is provided, use the current axis

    ax.plot(xtrain, ytrain, "x", label="Observations", color=cols[0], alpha=0.5)
    ax.fill_between(
        xtest.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.2,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(
        xtest,
        predictive_mean - 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(
        xtest,
        predictive_mean + 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    # Plot the inducing points (usually marked as triangles or other specific symbols)
    ax.plot(inducing_point, [0] * len(inducing_point), 'r^', markersize=6, label="Inducing points")  # Adjust markersize

    ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    return ax


