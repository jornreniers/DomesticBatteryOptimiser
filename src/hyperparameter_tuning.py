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
    config_full = features.run_full_time_resolution(df=df)

    # train the models to fit training data & compute scores on validation data
    return regression.run(config=config_full, figname_prefix="fullTime_")


def tune_hyper_params_fullTimeResolution():
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

    Each of these parameters can be set by the users in the InternalConfig.
    If they are set to None we do a grid search for the best value, otherwise we use the user-specified value.
    For the noise assumption, if the min and max are set the the same value, we use that; otherwise we do a grid search
    """

    # use manual selection of features or not
    # In total, there are 12 continuous features and 69 dummy options from 4 categorical features
    #   index: 48 options
    #   day of week: 7 options
    #   month: 12 options
    #   weekend: 2 options
    # with the manual selection, we have 9 continuous and 50 dummy-categorical values (index, weekend)
    if InternalConfig.features_fullResolution_forecast is None:
        manSel = [True, False]
        manual_features = [
            InternalConfig.colname_temperature_dry,
            InternalConfig.colname_daily_min_temperature,
            InternalConfig.colname_daily_temperature_below_zero,
            InternalConfig.colname_daily_temperature_below_five,
            InternalConfig.colname_daily_temperature_below_ten,
            InternalConfig.colname_daily_temperature_below_fifteen,
            InternalConfig.colname_daily_temperature_above_fifteen,
            InternalConfig.colname_daily_temperature_above_twenty,
            InternalConfig.colname_daily_temperature_above_twentyfive,
            InternalConfig.colname_weekend,  # boolean true or false
            InternalConfig.colname_period_index,  # integer eg 0-47
        ]
    else:
        manSel = [True]
        manual_features = InternalConfig.features_fullResolution_forecast

    # Fraction of features to keep at each stage
    if (InternalConfig.fullResolution_fraction_of_features_to_keep_kbest is None) | (
        InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv is None
    ):
        nfeatures_to_keep_kb = [1, 0.75, 0.5, 0.25]
        nfeatures_to_keep_rf = [1, 0.75, 0.5, 0.25]
    else:
        nfeatures_to_keep_kb = [
            InternalConfig.fullResolution_fraction_of_features_to_keep_kbest
        ]
        nfeatures_to_keep_rf = [
            InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv
        ]

    # Find the best fit
    errmin = np.inf
    manual_selection_optimal = True
    number_of_features_kbest = 0
    number_of_features_rfecv = 0

    # Store results from all iterations
    results = []

    grid_alpha = InternalConfig.lognoise_minimum is not InternalConfig.lognoise_maximim
    logger.info(
        f"Start tuning with {len(manSel)} options for features, {len(nfeatures_to_keep_kb)} and {len(nfeatures_to_keep_rf)} for number of features to keep, and grid search for noise {grid_alpha}"
    )

    for ms in manSel:
        if ms:
            InternalConfig.features_fullResolution_forecast = manual_features
        else:
            # Use all possible features, except the total daily consumption
            # which we wouldn't know in advance of predicting that day.
            # We could first use the daily forecaster to predict it and then
            # use it as a feature but that gets complicated.
            InternalConfig.features_fullResolution_forecast = (
                InternalConfig.features_categorical + InternalConfig.features_continuous
            )
            InternalConfig.features_fullResolution_forecast.remove(
                InternalConfig.colname_daily_consumption
            )

        # how many features to remove at which stage
        for nkbest in range(len(nfeatures_to_keep_kb)):
            for nrfecv in range(len(nfeatures_to_keep_rf)):
                InternalConfig.fullResolution_fraction_of_features_to_keep_kbest = (
                    nfeatures_to_keep_kb[nkbest]
                )
                InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv = (
                    nfeatures_to_keep_rf[nrfecv]
                )

                # Skip combos with more features in rfecv than in kbest since they have no effect
                if (
                    InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv
                    <= InternalConfig.fullResolution_fraction_of_features_to_keep_kbest
                ):
                    erri, noise, regressor = _train_fulltimeResolution()

                    # Store results from this iteration
                    results.append(
                        {
                            "ms": ms,
                            "kbest": InternalConfig.fullResolution_fraction_of_features_to_keep_kbest,
                            "rfecv": InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv,
                            "noise": noise,
                            "error": erri,
                        }
                    )

                    logger.info(
                        f"manual features {ms}, number kbest features {InternalConfig.fullResolution_fraction_of_features_to_keep_kbest}, number rfecv features {InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv} and noise {noise} resulted in error {erri}"
                    )

                    if erri < errmin:
                        errmin = erri
                        manual_selection_optimal = ms
                        number_of_features_kbest = InternalConfig.fullResolution_fraction_of_features_to_keep_kbest
                        number_of_features_rfecv = InternalConfig.fullResolution_fraction_of_features_to_keep_rfecv

                        logger.info(
                            f"Improved solution found with error {errmin}, manual features {manual_selection_optimal}, number kbest features {number_of_features_kbest}, number rfecv features {number_of_features_rfecv} and noise {noise}"
                        )

    # Print all results after loops complete
    logger.info("\n" + "=" * 80)
    logger.info("Summary of hyperparameter tuning:")
    for i, result in enumerate(results, 1):
        logger.info(
            f"Iteration {i}: manual features {result['ms']}, number kbest features {result['kbest']}, "
            f"number rfecv features {result['rfecv']} and noise {result['noise']:.6f} resulted in error {result['error']:.6f}"
        )
    logger.info("=" * 80 + "\n")

    logger.info(
        f"Optimal fit found with error {errmin}, manual features {manual_selection_optimal}, number kbest features {number_of_features_kbest}, number rfecv features {number_of_features_rfecv} and noise {noise}"
    )
