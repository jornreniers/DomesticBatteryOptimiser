from config.InternalConfig import InternalConfig
import logging

from src.hyperparameter_tuning import (
    valid_hyperparams,
    tune_hyper_params_fullTimeResolution,
)
from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression

logger = logging.getLogger()


def run_daily_total():
    logger.info("Start training for forecasting total daily consumption")
    # read data
    df = data_ingestor.run()

    # select features
    config_day = features.run_daily_total(df=df)

    # train the models to fit training data & compute scores on validation data
    regression.run(
        config=config_day,
        plotfolder=InternalConfig.plot_folder + "/fitting",
        figname_prefix="daily_",
    )


def run_full_time_resolution():
    logger.info(
        "Start training for forecasting consumption at the full time resolution"
    )
    # read data
    df = data_ingestor.run()

    # select features
    config_full = features.run_full_time_resolution(df=df)

    # train the models to fit training data & compute scores on validation data
    regression.run(
        config=config_full,
        plotfolder=InternalConfig.plot_folder + "/fitting",
        figname_prefix="fullTime_",
    )


def main():

    # Set the logging level to info
    logging.basicConfig(
        level=logging.INFO,  # Global minimum logging level DEBUG, INFO, WARNING, ERROR
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Uncomment what you want to run
    # run_daily_total()

    # TODO best fit is
    # 2026-02-19 16:52:01 [INFO] root: Optimal fit found with error 0.6198210176868121, manual features False, number kbest features 0.5, number rfecv features 0.5 and noise -1
    # so make an option to run without manual selection of features. atm you can only do manual or tuning, not select from all.
    # then set that as default and move on.

    if valid_hyperparams():
        run_full_time_resolution()
    else:
        tune_hyper_params_fullTimeResolution()


if __name__ == "__main__":
    main()
