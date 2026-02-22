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

    # If all parameters are known, run that one
    # Results are plotted along the way
    if InternalConfig.lognoise_minimum == InternalConfig.lognoise_maximim:
        err, f_in_std_t, f_in_std_v, y_pred, y_std, forecaster = (
            gaussian_process_regression.gaussian_process_regression(
                config=config,
                plotfolder=plotfolder,
                figname_prefix=figname_prefix,
                noise=InternalConfig.lognoise_minimum,
            )
        )

        logger.info(
            f"Training completed with MAPE {err:.3f}, and {f_in_std_v * 100:.2f}% of data points within +- 1 std in validation data"
        )

        noise = InternalConfig.lognoise_minimum
        if InternalConfig.plot_level >= 2:
            prefix = figname_prefix + f"gaussian_process_optimal_alpha{noise}_"
            score_regression.score_and_plot_trained_model(
                config=config,
                y_pred=y_pred,
                y_std=y_std,
                alpha=pow(10, noise),
                plotfolder=plotfolder,
                figname_prefix=prefix,
                ploton=True,
            )
    # Otherwise tune hyperparams
    else:
        err, noise, forecaster = tuning.tune_gpr_hyperparam(
            config=config, plotfolder=plotfolder, figname_prefix=figname_prefix
        )

    return err, noise, forecaster
