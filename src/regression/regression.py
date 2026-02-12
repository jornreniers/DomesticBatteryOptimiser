import logging
import os

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration
from src.regression import gaussian_process_regression
from src.regression import score_regression
from src.regression import tuning

logger = logging.getLogger()


def run(
    config: FeatureConfiguration, plotfolder: str, figname_prefix: str
) -> tuple[float, float, gaussian_process.GaussianProcessRegressor]:

    logger.info("Start regression")

    if not (os.path.exists(plotfolder)):
        os.makedirs(plotfolder)

    if InternalConfig.lognoise_minimum == InternalConfig.lognoise_maximim:
        err, y_pred, y_std, forecaster = (
            gaussian_process_regression.gaussian_process_regression(
                config=config,
                plotfolder=plotfolder,
                figname_prefix=figname_prefix,
                noise=InternalConfig.lognoise_minimum,
            )
        )
        noise = InternalConfig.lognoise_minimum
        if InternalConfig.plot_level >= 2:
            prefix = figname_prefix + f"gaussian_process_optimal_alpha{noise}_"
            score_regression.score_and_plot_trained_model(
                config=config,
                y_pred=y_pred,
                y_std=y_std,
                plotfolder=plotfolder,
                figname_prefix=prefix,
                ploton=True,
            )
    else:
        err, noise, forecaster = tuning.tune_gpr_hyperparam(
            config=config, plotfolder=plotfolder, figname_prefix=figname_prefix
        )

    return err, noise, forecaster
