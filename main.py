import math
import logging

import numpy as np
from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression

logger = logging.getLogger()


def _train_fulltimeResolution() -> tuple[
    float, float, gaussian_process.GaussianProcessRegressor
]:
    # read data
    df = data_ingestor.run()

    # select features
    config_day, config_full = features.run(df=df)

    # train the models to fit training data & compute scores on validation data
    return regression.run(config=config_full, figname_prefix="fullTime_")


def _tune_hyper_params_fullTimeResolution():
    """
    There are broadly 4 hyperparameters used in fitting the regression model
    (1) manual specification of relevant features
        InternalConfig.features_daily_forecast and InternalConfig.features_fullResolution_forecast
    (2) number of features removed using kbest in feature selection (or rather, number kept after using it)
        InternalConfig.daily_min_number_of_features_kbest and InternalConfig.fullResolution_min_number_of_features_kbest
    (3) number of features removed using co-linear remover and rfecv in feature selection (or rather, number kept after using it)
        InternalConfig.daily_min_number_of_features_rfecv and InternalConfig.fullResolution_min_number_of_features_rfecv
    (4) noise assumption of the guassian process. The optimal value is found with a simple grid-search in the range
        10^InternalConfig.lognoise_minimum to 10^InternalConfig.lognoise_maximim

    We can't use pre-implemented tools like SKlearn's GridSearchCV because the parameters are spread between
    different parts of the code
    """

    # use manual selection of features or not
    manSel = [True, False]
    manual_features = InternalConfig.features_fullResolution_forecast

    # Fraction of features to keep at each stage
    nfeatures_to_keep = [1, 0.75, 0.5, 0.25, 0.1]

    # Find the best fit
    errmin = np.inf
    manual_selection_optimal = True
    number_of_features_kbest = 0
    number_of_features_rfecv = 0

    for ms in manSel:
        if ms:
            InternalConfig.features_fullResolution_forecast = manual_features
        else:
            InternalConfig.features_fullResolution_forecast = (
                InternalConfig.features_categorical + InternalConfig.features_continuous
            )

        # how many features to remove at which stage
        nf = len(InternalConfig.features_fullResolution_forecast)
        for nkbest in range(len(nfeatures_to_keep)):
            for nrfecv in range(nkbest, len(nfeatures_to_keep), 1):
                InternalConfig.fullResolution_min_number_of_features_kbest = math.ceil(
                    nfeatures_to_keep[nkbest] * nf
                )
                InternalConfig.fullResolution_min_number_of_features_rfecv = math.ceil(
                    nfeatures_to_keep[nrfecv] * nf
                )

                logger.info(
                    f"Start fitting with manaul features {ms}, number kbest features {InternalConfig.fullResolution_min_number_of_features_kbest}, number rfecv features {InternalConfig.fullResolution_min_number_of_features_rfecv}"
                )
                erri, noise, regressor = _train_fulltimeResolution()

                if erri < errmin:
                    errmin = erri
                    manual_selection_optimal = ms
                    number_of_features_kbest = (
                        InternalConfig.fullResolution_min_number_of_features_kbest
                    )
                    number_of_features_rfecv = (
                        InternalConfig.fullResolution_min_number_of_features_rfecv
                    )

                    logger.info(
                        f"Improved solution found with error {errmin}, manaul features {manual_selection_optimal}, number kbest features {number_of_features_kbest}, number rfecv features {number_of_features_rfecv} and noise {noise}"
                    )

    logger.info(
        f"Optimal fit found with error {errmin}, manaul features {manual_selection_optimal}, number kbest features {number_of_features_kbest}, number rfecv features {number_of_features_rfecv} and noise {noise}"
    )


def main():

    logging.basicConfig(
        level=logging.DEBUG,  # Global minimum logging level
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _tune_hyper_params_fullTimeResolution()

    # # read data
    # df = data_ingestor.run()

    # # select features
    # config_day, config_full = features.run(df=df)

    # # train the models to fit training data & compute scores on validation data
    # logger.debug("Start fitting daily total consumption")
    # regression.run(config=config_day, figname_prefix="daily_")
    # logger.debug("Start fitting full time resolution data")
    # regression.run(config=config_full, figname_prefix="fullTime_")

    # # TODO compare forecast for daily demand with sum of full-time-resolution forecast


if __name__ == "__main__":
    main()
