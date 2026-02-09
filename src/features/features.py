from config.InternalConfig import InternalConfig
import polars as pl
from typing import cast
from src.features import (
    add_features,
    daily_averages,
    process_features,
    feature_selection,
    data_analyser,
)
from src.features.feature_configuration import FeatureConfiguration


def run(df: pl.DataFrame) -> tuple[FeatureConfiguration, FeatureConfiguration]:
    """
    Take the dataframe with the raw data, and add columns with features to it.
    the FeatureConfiguration will keep track of which features should be used
    """

    # Add all features and prepare them for learning
    dfl = add_features.run(df=df.lazy())
    # Collect can return a df or a InProcessQuery (streaming implementation)
    # so ty complains. Add an explicit cast to make it happy
    df_with_features = cast(pl.DataFrame, dfl.collect())

    # Plot full-time-resolution consumption vs various metrics
    # useful to explore the data and visually inspect the effect
    # of different features and metrics
    data_analyser.run(df=df_with_features)

    # Select features to predict total daily consumption
    dfl_day = daily_averages.run(df=dfl)
    daily_config = FeatureConfiguration(
        df=cast(pl.DataFrame, dfl_day.collect()),
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=False,
    )
    process_features.run(config=daily_config)
    daily_config.set_training_data_filter()
    feature_selection.run(config=daily_config, figname_prefix="daily_")

    # Select features to predict full-time-resolution consumption
    # TODO reconsider features, we don't want to redo the work from total daily consumption
    # TODO use insights from data_analyser
    full_config = FeatureConfiguration(
        df=df_with_features,
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=True,
    )
    process_features.run(config=full_config)
    full_config.set_training_data_filter()
    feature_selection.run(config=full_config, figname_prefix="fullTime_")

    return daily_config, full_config
