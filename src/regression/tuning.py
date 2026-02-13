import logging

import numpy as np
from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression import gaussian_process_regression
from src.regression import score_regression


logger = logging.getLogger()


def tune_gpr_hyperparam(
    config: FeatureConfiguration, plotfolder: str, figname_prefix: str
) -> tuple[float, float, gaussian_process.GaussianProcessRegressor]:
    """
    Find the value of alpha that minimises the error between forecast and data
    in the validation data set.

    This could be done with SKlearn's GridSearchCV in the future
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    alpha_power_range = np.linspace(
        InternalConfig.lognoise_minimum, InternalConfig.lognoise_maximim, 10
    )
    errmin = np.inf
    alpha = np.inf
    forecaster = gaussian_process.GaussianProcessRegressor()
    y_pred = np.array([])
    y_std = np.array([])
    for ap in alpha_power_range:
        err, f_in_std_t, f_in_std_v, y_pred, y_std, fi = (
            gaussian_process_regression.gaussian_process_regression(
                config=config,
                plotfolder=plotfolder,
                figname_prefix=figname_prefix,
                noise=ap,
            )
        )

        logger.info(
            f"Tuning with {ap} measurement noise. MAPE {err:.3f}, and {f_in_std_v * 100:.2f}% in +- 1 std for validation or {f_in_std_t * 100:.2f}% for training"
        )

        # TODO there are two options to pick from:
        # (1) minimise forecasting error
        # (2) minimise uncertainty on forecasting
        # We need a mix of both, the former results in quite a large uncertainty
        # while the latter gets a fit which is good during the day (when demand is constant)
        # but isn't good at predicting peaks (which are just a few data points)

        # OPTION 1: minimise the forecasting error
        # if err < errmin:
        #     errmin = err
        #     alpha = ap
        #     forecaster = fi
        # OPTION 2: smallest alpha that has 68% of data points in +- 1 std
        # loop goes from small (~0%) to large (100%) so as soon as we find one, stop
        if f_in_std_v >= 0.68:
            errmin = err
            alpha = ap
            forecaster = fi
            break

    logger.info(
        f"The optimal fit has noise exponent {alpha}, resulting in an error of {errmin}"
    )

    # Make all the plots and write results to csv
    # intermediate plots (for all attempted values of alpha) are only made for plotlevel >= 3
    if InternalConfig.plot_level >= 2:
        prefix = figname_prefix + f"gaussian_process_optimal_alpha{alpha}_"
        score_regression.score_and_plot_trained_model(
            config=config,
            y_pred=y_pred,
            y_std=y_std,
            alpha=pow(10, ap),
            plotfolder=plotfolder,
            figname_prefix=prefix,
            ploton=True,
        )

    return errmin, alpha, forecaster
