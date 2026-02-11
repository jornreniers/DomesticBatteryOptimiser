import logging

import polars as pl
from typing import cast

from config.InternalConfig import InternalConfig
from src.features import (
    add_features,
    daily_averages,
    process_features,
    feature_selection,
    full_resolution_data_analyser,
)
from src.features.feature_configuration import FeatureConfiguration

logger = logging.getLogger()


def _feature_selection_for_daily_totals(dfl: pl.LazyFrame) -> FeatureConfiguration:
    """
    Perform feature selection for when we want to forecast the total daily consumption.
    """

    # Groupby date so we get daily totals
    dfl_day = daily_averages.run(df=dfl)

    # Set up the configuration and manually
    # filter out unneeded features (keep those defined in InternalConfig.features_daily_forecast)
    daily_config = FeatureConfiguration(
        df=cast(pl.DataFrame, dfl_day.collect()),
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=False,
        list_of_features=InternalConfig.features_daily_forecast,
    )
    process_features.run(config=daily_config)
    daily_config.set_training_data_filter()
    feature_selection.run(config=daily_config, figname_prefix="daily_")

    return daily_config


def _feature_selection_for_full_time_resolution(
    df: pl.DataFrame,
) -> FeatureConfiguration:
    """
    Perform feature selection for when we want to forecast the consumption at each point in time.
    """
    full_config = FeatureConfiguration(
        df=df,
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=True,
        list_of_features=InternalConfig.features_fullResolution_forecast,
    )
    process_features.run(config=full_config)
    full_config.set_training_data_filter()
    feature_selection.run(config=full_config, figname_prefix="fullTime_")

    return full_config


def run_daily_total(
    df: pl.DataFrame,
) -> FeatureConfiguration:
    """
    Take the dataframe with the raw data, and add columns with features to it.
    the FeatureConfiguration will keep track of which features should be used
    """
    logger.info("Start feature selection")

    # Add all features and prepare them for learning
    dfl = add_features.run(df=df.lazy())

    # Select features to predict total daily consumption
    daily_config = _feature_selection_for_daily_totals(dfl=dfl)

    return daily_config


def run_full_time_resolution(
    df: pl.DataFrame,
) -> FeatureConfiguration:
    """
    Take the dataframe with the raw data, and add columns with features to it.
    the FeatureConfiguration will keep track of which features should be used
    """
    logger.info("Start feature selection")

    # Add all features and prepare them for learning
    dfl = add_features.run(df=df.lazy())
    # Collect can return a df or a InProcessQuery (streaming implementation)
    # so ty complains. Add an explicit cast to make it happy
    df_with_features = cast(pl.DataFrame, dfl.collect())

    # Plot full-time-resolution consumption vs various metrics
    # useful to explore the data and visually inspect the effect
    # of different features and metrics
    full_resolution_data_analyser.run(df=df_with_features)

    # select features to predict full time resolution
    full_config = _feature_selection_for_full_time_resolution(df=df_with_features)

    return full_config
