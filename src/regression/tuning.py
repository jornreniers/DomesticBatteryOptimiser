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
        err, y_pred, y_std, fi = (
            gaussian_process_regression.gaussian_process_regression(
                config=config,
                plotfolder=plotfolder,
                figname_prefix=figname_prefix,
                noise=ap,
            )
        )
        if err < errmin:
            errmin = err
            alpha = ap
            forecaster = fi

    logger.info(
        f"The optimal fit has noise exponent {alpha}, resulting in an error of {errmin}"
    )

    # Make all the plots and write results to csv
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
