from plotly.offline.offline import plot
import logging

import numpy as np

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression import score_regression

logger = logging.getLogger()


def gaussian_process_regression(
    config: FeatureConfiguration, plotfolder: str, figname_prefix: str, noise: float
) -> tuple[float, np.ndarray, np.ndarray, gaussian_process.GaussianProcessRegressor]:
    """
    Fit the consumption with a gaussian process
    the advantage is that this type of model allows "noise" in measurement data
    which can reflect uncertainty or random variation.
    After all, electricity consumption will always depend on random factors
    (when do you iron, do the laundry, use the oven to bake a cake, etc).
    We want to avoid overfitting the data we have, so by setting the correct
    noise threshold in the gaussian process, it will aim to fit the average
    expected consumption for the conditions of that day.

    A second advantage is that it gives some uncertainty estimate,
    however, this should not be fully relied upon depending on the amount
    of training data.

    Returns the MAPE on the validation data
    """
    # gaussian processes: https://scikit-learn.org/stable/modules/gaussian_process.html

    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()

    # Fit the training data, and predict all data
    # for noise in [-1]:  # [-1, -0.5, 0, 0.5]:
    reg = gaussian_process.GaussianProcessRegressor(
        normalize_y=True, alpha=pow(10, noise)
    ).fit(x, y)

    # Evaluate for all data points. This takes a relative long time
    # so only do it once and pass the results around
    y_pred, y_std = reg.predict(
        config.df.select(config.get_features()).to_numpy(), return_std=True
    )

    # # The uncertainty the GP returns is for what it thinks is the "true" value
    # # after accounting for "measurement error or noise".
    # # What we supplied as "noise" is however not measurement error but
    # # reflects random variations in the actual value. When forecasting values, we should
    # # therefore add it to the uncertainty, so we can check whether the
    # # value (ie what the GP thinks is truth + noise) is within
    # # the forecasted range.
    # # Ie (y_pred, y_std) is the distribution of the "true" value
    # # while we are interested in the "noisy" value
    # # so we add the "noise" to the standard deviation
    # y_std = y_std + pow(10, noise)

    # Score how well the fit went, plot if desired
    prefix = figname_prefix + f"gaussian_process_optimal_alpha{noise}_"
    err_t, err_v = score_regression.score_and_plot_trained_model(
        config=config,
        y_pred=y_pred,
        y_std=y_std,
        plotfolder=plotfolder,
        figname_prefix=prefix,
        ploton=InternalConfig.plot_level >= 3,
    )
    logger.debug(
        f"MAPE on training data is {err_t} and on validation data {err_v} for fitting on {prefix}"
    )

    return err_v, y_pred, y_std, reg
